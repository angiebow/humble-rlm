"""Direct mlx-lm generation with per-token entropy + attention introspection.

DRAGIN's RIND needs H_i (full-vocab entropy) and a_max(i) (self-attention from
later tokens back to t_i, last transformer layer) at every generated token.
Neither is obtainable through an OpenAI-compatible REST API -- mlx_lm.server only
returns the chosen token's logprob, never attention weights, and the paper's own
Limitations (§7) says as much: "our method is not applicable to certain APIs that
do not provide access to the self-attention scores." So this module talks to
mlx-lm directly, in-process, bypassing litellm and the REST bridge entirely for
the root model. (The worker model, used only as a plain answer generator over an
already-selected passage, stays on the regular litellm path -- see harness.py.)

Two adaptations versus the paper, both specific to Qwen3.5-35B-A3B:

1. HYBRID ATTENTION. mlx_lm.models.qwen3_5 / qwen3_5_moe is not a plain
   Transformer: most decoder layers run GatedDeltaNet, a recurrent/SSM mechanism
   with no pairwise attention matrix at all (DecoderLayer.is_linear). Only every
   ``full_attention_interval``-th layer (config default 4) runs real softmax
   self-attention (qwen3_next.Qwen3NextAttention). The paper's "last transformer
   layer" (footnote 2) is reinterpreted here as the last layer that actually has
   one -- see _find_last_attention_layer.
2. UNFUSED ATTENTION. mlx-lm's attention path calls the fused
   ``mx.fast.scaled_dot_product_attention`` kernel, which never materializes the
   softmax(QK^T) matrix, so it can't be read after the fact. _patch_attention_layer
   replaces ``type(layer.self_attn).__call__`` at the CLASS level (Python looks up
   dunder methods on the type, not the instance, for implicit calls like
   ``attn(x, ...)`` -- an instance-level patch is silently never invoked; an
   earlier version of this got that wrong, see _patch_attention_layer's
   docstring), guarded so only the ONE targeted instance actually diverges from
   the true original -- an unfused reimplementation (manual QK^T -> causal mask
   -> softmax -> matmul-V) that reproduces
   mlx_lm.models.qwen3_next.Qwen3NextAttention.__call__ line for line except for
   that one substitution -- re-check against mlx-lm's source if this starts
   failing verify_against_fused(), since it will drift if mlx-lm's attention
   implementation changes.

IMPORTANT: do not drive this with ``mlx_lm.generate_step``. That generator
computes step n+1 via ``mx.async_eval`` *before* yielding step n (latency hiding),
so a naive "read the probe buffer after each yield" wrapper would silently
capture the WRONG step's attention -- an off-by-one that would look like it works
(no crash, plausible-looking numbers) while being wrong. generate_with_probe
below runs its own synchronous, one-step-at-a-time loop instead, with no
speculative lookahead, specifically so the probe buffer always matches the token
just yielded.

Run verify_against_fused() once on real hardware before trusting any RIND score
this module produces -- it was written and reviewed against mlx-lm's source but
never executed (this development environment has no mlx / Apple GPU access).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Tuple


@dataclass
class ProbedToken:
    text: str
    token_id: int
    entropy: float
    attn_row: List[float]  # this token's own attention back to positions [0, i)


def _find_last_attention_layer(model: Any) -> Any:
    """Highest-index decoder layer with real self-attention (is_linear False)."""
    layers = model.layers
    for layer in reversed(layers):
        if not getattr(layer, "is_linear", False):
            return layer
    raise RuntimeError(
        "No full-attention layer found on this model -- unexpected for the "
        "Qwen3.5 family (every full_attention_interval-th layer should have one; "
        "check model.args.text_config['full_attention_interval'])."
    )


def _patch_attention_layer(
    layer: Any, on_probs: Callable[[Any], None]
) -> Tuple[type, Callable[..., Any]]:
    """Replace ``type(layer.self_attn).__call__`` with an unfused equivalent
    that reports the softmax attention matrix via ``on_probs`` before returning
    exactly the output the fused kernel would have -- but ONLY for the specific
    ``layer.self_attn`` instance; every other instance of the same class (other
    full-attention layers, or a second model loaded in the same process, as
    verify_against_fused's own internal load does) falls through unchanged to
    the true original implementation.

    Patching the class rather than the instance is deliberate, not incidental:
    ``attn.__call__ = fn`` sets an INSTANCE attribute, but Python looks up
    dunder methods like ``__call__`` on the TYPE when the implicit call syntax
    (``attn(x, ...)``) is used -- the instance attribute is silently never
    consulted. An earlier version of this function patched the instance and
    passed verify_against_fused() with max|delta logits| == 0, which in
    hindsight was itself the tell: an independently-derived unfused
    reimplementation essentially never reproduces a fused kernel's output
    bit-for-bit, only "the patch never actually ran" does. Caught for real by
    generate_with_probe's own captured-nothing guard on first real use.

    Returns (attn_cls, original_call) so the caller can restore exactly:
    ``attn_cls.__call__ = original_call``.

    Computes the real output via the SAME fused kernel the original
    implementation uses (mlx_lm.models.base.scaled_dot_product_attention,
    O(L) memory) rather than reimplementing that computation by hand. Only the
    softmax attention weights for the LAST query position are computed
    manually (O(S) memory) -- the only row generate_with_probe ever reads
    (``probs[0, :, -1, :]``, since RIND/QFS only ever care about the token
    just generated). An earlier version manually recomputed the FULL L x S
    attention matrix to get that one row, which is exactly the O(L*S)
    materialization the fused kernel exists to avoid: for BrowseComp-Plus's
    multi-thousand-token documents during the prefill step (L == S == prompt
    length), that allocation reliably exceeded even Metal's ~167GB buffer
    limit (observed: "Attempting to allocate 211645384832 bytes" on a real
    run). No causal mask is needed for the last-position row: in a causal
    sequence the LAST query position is always allowed to attend to every key
    that exists (nothing comes after it to mask out).

    Reimplements the READOUT PATH only; mlx_lm.models.qwen3_next.
    Qwen3NextAttention.__call__ itself is delegated to unchanged (via
    scaled_dot_product_attention) for the actual output.
    """
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    attn = layer.self_attn
    attn_cls = type(attn)
    original_call = attn_cls.__call__

    def patched_call(self, x, mask=None, cache=None):
        if self is not attn:
            return original_call(self, x, mask=mask, cache=cache)

        B, L, D = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        n_kv_heads = keys.shape[1]
        n_repeats = self.num_attention_heads // n_kv_heads
        S = keys.shape[2]
        q_last = queries[:, :, -1:, :]  # (B, H, 1, D) -- only row ever read

        # Stays float32: this is never fed back into the model (only read out
        # for RIND/QFS via on_probs -> generate_with_probe -> numpy), and
        # mx.array's bfloat16 dtype has no numpy buffer-protocol equivalent --
        # np.asarray() on a bf16 array raises ValueError: 'bfloat16' is not a
        # valid PEP 3118 buffer format string. Casting back to the model's
        # native dtype here (an earlier version did) reintroduces that crash.
        if n_repeats > 1:
            q_last4 = q_last.reshape(B, n_kv_heads, n_repeats, 1, -1)
            k4 = mx.expand_dims(keys, 2)
            last_scores = (q_last4 * self.scale) @ k4.swapaxes(-1, -2)
            last_probs = mx.softmax(last_scores.astype(mx.float32), axis=-1)
            last_probs = last_probs.reshape(B, self.num_attention_heads, 1, S)
        else:
            last_scores = (q_last * self.scale) @ keys.swapaxes(-1, -2)
            last_probs = mx.softmax(last_scores.astype(mx.float32), axis=-1)

        on_probs(last_probs)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    attn_cls.__call__ = patched_call
    return attn_cls, original_call


def load_dragin_model(model_path: str) -> Tuple[Any, Any]:
    """Load a model + tokenizer for direct (non-REST) generation with mlx-lm."""
    from mlx_lm import load

    return load(model_path)


def generate_with_probe(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    prefill_step_size: int = 2048,
) -> Iterator[ProbedToken]:
    """Synchronous, one-step-at-a-time generation yielding entropy + attention
    for every token. Restores the original attention layer on exit (including
    early ``.close()``/break) via try/finally.

    Prefill is chunked at ``prefill_step_size`` tokens, mirroring
    mlx_lm.generate_step's own behavior -- BrowseComp-Plus documents can run
    to 80k+ tokens (observed: 81305 for one real example), and a single
    monolithic forward pass over the whole prompt is what actually crashed
    the first live run here (silently, no Python traceback -- a native
    Metal-level fault, not something the attention patch's own memory fix
    caught). verify_against_fused's ~12-token test prompt never exercises
    this path, which is why it always passes regardless. Only the FINAL
    chunk's forward pass matters for sampling/probing -- every earlier
    chunk's only job is to populate the KV cache; its own logits and
    attention row are interior-prompt noise, not the signal RIND cares about
    (always the token about to be generated), so they're discarded.
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    probe_layer = _find_last_attention_layer(model)
    captured: List[Any] = [None]

    def _on_probs(probs_full: Any) -> None:
        captured[0] = probs_full

    attn_cls, original_call = _patch_attention_layer(probe_layer, _on_probs)
    sampler = make_sampler(temp=temperature)
    try:
        prompt_tokens = mx.array(tokenizer.encode(prompt))
        prompt_cache = make_prompt_cache(model)

        def step(input_tokens: Any) -> Tuple[Any, Any]:
            logits = model(input_tokens[None], cache=prompt_cache).astype(mx.float32)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            token = sampler(logprobs)
            mx.eval(token, logprobs)
            return token, logprobs

        remaining = prompt_tokens
        while len(remaining) > 1:
            n_chunk = min(prefill_step_size, len(remaining) - 1)
            out = model(remaining[:n_chunk][None], cache=prompt_cache)
            mx.eval(out)
            remaining = remaining[n_chunk:]
            mx.clear_cache()

        token, logprobs = step(remaining)
        n = 0
        while n < max_tokens:
            probs = captured[0]
            if probs is None:
                raise RuntimeError(
                    "attention probe captured nothing -- the patched layer's "
                    "__call__ never ran; check _find_last_attention_layer picked "
                    "a layer actually on the forward path."
                )
            s_total = probs.shape[-1]
            row = np.asarray(probs[0, :, -1, : s_total - 1].mean(axis=0)).tolist()
            lp = np.asarray(logprobs[0])
            entropy = float(-(np.exp(lp) * lp).sum())
            tok_id = int(token.item())
            text = tokenizer.decode([tok_id])
            yield ProbedToken(text=text, token_id=tok_id, entropy=entropy, attn_row=row)
            if tok_id in tokenizer.eos_token_ids:
                return
            n += 1
            if n >= max_tokens:
                return
            token, logprobs = step(token.reshape(1))
    finally:
        attn_cls.__call__ = original_call


def verify_against_fused(
    model_path: str,
    prompt: str = "The quick brown fox jumps over the lazy dog. It then",
    atol: float = 1e-3,
    model: Any = None,
    tokenizer: Any = None,
) -> bool:
    """One-time sanity check: does the patched (unfused) attention layer produce
    the SAME end-to-end logits as the original fused kernel, given the same
    input? Run this once on real hardware -- and make it pass -- before trusting
    any RIND score from generate_with_probe. Not run automatically; call it from
    a script or REPL on the machine that actually has mlx + the model weights.

    Pass an already-loaded ``model``/``tokenizer`` (e.g. from a caller that
    also needs them for the real run right after) to avoid loading a second
    full copy of a multi-billion-parameter model just for this check --
    experiments/run_dragin.py does this. Loads its own if omitted, for
    standalone use from a REPL.
    """
    import mlx.core as mx

    if model is None or tokenizer is None:
        model, tokenizer = load_dragin_model(model_path)
    tokens = mx.array(tokenizer.encode(prompt))[None]

    from mlx_lm.models.cache import make_prompt_cache

    baseline_logits = model(tokens, cache=make_prompt_cache(model))
    mx.eval(baseline_logits)

    layer = _find_last_attention_layer(model)
    attn_cls, original_call = _patch_attention_layer(layer, lambda _probs: None)
    try:
        patched_logits = model(tokens, cache=make_prompt_cache(model))
        mx.eval(patched_logits)
    finally:
        attn_cls.__call__ = original_call

    diff = float(mx.abs(baseline_logits - patched_logits).max())
    ok = diff < atol
    status = "PASS" if ok else "FAIL"
    print(
        f"[dragin_rlm] unfused-attention patch check: max|delta logits| = "
        f"{diff:.6g} ({status}, atol={atol})"
    )
    return ok
