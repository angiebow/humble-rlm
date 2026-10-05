"""Shows that MLX's random state is per-thread: seeding in a different thread is ignored and a fresh thread replays the
same draws. This is the root cause of the mlx_lm.server 0.31.3 problem (identical samples on every request).
Run: python rng_test.py   (uses the cached Qwen3.5-2B; expects A/B/D to vary and C/E to be identical)"""
import threading

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

model, tok = load("mlx-community/Qwen3.5-2B-bf16")
msgs = [{"role": "user", "content": "Pick a random number between 1 and 1000. Reply with just the number."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def gen():
    return generate(model, tok, prompt=prompt, max_tokens=6, sampler=make_sampler(temp=1.0)).strip()


def in_thread(fn):
    out = []

    def run():
        with mx.stream(mx.new_stream(mx.default_device())):
            out.append(fn())

    t = threading.Thread(target=run)
    t.start()
    t.join()
    return out[0]


print("A main, unseeded x5     ", [gen() for _ in range(5)])
print("B main, seed 0..4       ", [(mx.random.seed(s), gen())[1] for s in range(5)])
print("C thread, unseeded x5   ", [in_thread(gen) for _ in range(5)])
print("D thread, seed in thread", [in_thread(lambda s=s: (mx.random.seed(s), gen())[1]) for s in range(5)])
print("E seed main, gen thread ", [(mx.random.seed(s), in_thread(gen))[1] for s in range(5)])
