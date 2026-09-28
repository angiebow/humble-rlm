"""Dataset utilities: JSONL I/O, deterministic splits, gold-evidence fingerprints.

Processed example schema (one JSON object per line):
  qid, dataset, split, query, context, gold_answer,
  gold_fingerprints: [[str, ...], ...]  one list per gold evidence item
  gold_mode: "any" | "all"              which items are needed to be "sufficient"
  evidence_fingerprints: [[str, ...]]   broader relevant set (for evidence precision)
  length_bucket, complexity, context_tokens, meta
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def assign_split(qid: str, val_fraction: float = 0.3) -> str:
    """Deterministic, seed-free split: the same qid always lands in the same split."""
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "test"


def window_fingerprints(context: str, start: int, end: int, pad: int = 40) -> List[str]:
    """Unique-ish substrings around context[start:end] that survive chunk boundaries."""
    core = context[start:end]
    after = context[start: min(len(context), end + pad)]
    before = context[max(0, start - pad): end]
    return [fp for fp in {after, before} if fp] or [core]


def doc_fingerprints(text: str, n: int = 3, width: int = 100) -> List[str]:
    """Snippets from inside a document, so a partial read of the doc still counts."""
    text = text.strip()
    if len(text) <= width:
        return [text] if text else []
    positions = [int(len(text) * (i + 1) / (n + 1)) for i in range(n)]
    return [text[p: p + width] for p in positions]


# --------------------------------------------------------------------- BABILong
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
MOVE_VERBS = r"(went|journeyed|travelled|traveled|moved|went back)"
TAKE_VERBS = r"(picked up|got|grabbed|took)"
DROP_VERBS = r"(dropped|discarded|put down|left)"


def babi_facts(zero_k_input: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(zero_k_input.strip()) if s.strip()]


def _mentions(sentence: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE) is not None


def babilong_gold_facts(task: str, question: str, facts: List[str], target: str
                        ) -> Tuple[List[int], bool]:
    """Indices of the supporting facts for qa1/qa2, and whether the rule succeeded.

    qa1 "Where is Mary?": the last fact mentioning Mary.
    qa2 "Where is the football?": the last take/drop of the object by its holder,
        plus the holder's last movement before a drop (or overall, if still held).
    """
    q = question.strip().rstrip("?")
    if task == "qa1":
        m = re.match(r"where is (\w+)", q, re.IGNORECASE)
        if not m:
            return [], False
        person = m.group(1)
        idx = [i for i, f in enumerate(facts) if _mentions(f, person)]
        if not idx:
            return [], False
        return [idx[-1]], _mentions(facts[idx[-1]], target)

    if task == "qa2":
        m = re.match(r"where is the (\w+)", q, re.IGNORECASE)
        if not m:
            return [], False
        obj = m.group(1)
        last_obj = None
        for i, f in enumerate(facts):
            if _mentions(f, obj) and re.search(TAKE_VERBS + "|" + DROP_VERBS, f):
                last_obj = i
        if last_obj is None:
            return [], False
        holder = facts[last_obj].split()[0]
        dropped = re.search(DROP_VERBS, facts[last_obj]) is not None
        limit = last_obj if dropped else len(facts)
        moves = [i for i in range(limit)
                 if _mentions(facts[i], holder) and re.search(MOVE_VERBS, facts[i])]
        if not moves:
            return [last_obj], False
        loc = moves[-1]
        return sorted({last_obj, loc}), _mentions(facts[loc], target)

    return [], False


def align_facts(context: str, facts: List[str]) -> Optional[List[int]]:
    """Character offset of each fact in the long context, matched in order.

    Returns None if any fact cannot be found (the long sample is not aligned with
    its 0k version); such samples are skipped rather than mislabelled.
    """
    positions, cursor = [], 0
    for fact in facts:
        pos = context.find(fact, cursor)
        if pos < 0:
            return None
        positions.append(pos)
        cursor = pos + len(fact)
    return positions
