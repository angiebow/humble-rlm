"""Build data/processed/multidata_probe.jsonl -- one question from each of
five datasets with very different context shapes, to probe which task
DRAGIN-RLM is actually suited to (scripts/run_multidata_suitability.py runs
DRAGIN-RLM over this file afterward).

    python scripts/build_multidata_probe.py

Datasets and why each is here:
  - BABILong      synthetic long-context multi-hop (needle-in-haystack style)
  - BrowseComp-Plus  real web-scale QA, gold+evidence+hard-negative documents
  - HotpotQA      short multi-hop QA (10 candidate paragraphs, ~1-2K tokens
                   total) -- the one SHORT-context dataset here, a deliberate
                   contrast: does DRAGIN-RLM's design assume long context?
  - OOLONG        long-context aggregation over D&D actual-play transcripts
                   (oolongbench/oolong-real, config toy_dnd) -- ~40-50K tokens,
                   questions require counting/aggregating across the whole
                   transcript, not single-passage lookup
  - RepoQA        "searching needle function" in a real codebase (evalplus/
                   repoqa) -- precise code search over ~400K+ chars of source,
                   the closest real benchmark to "Repo-Level CodeQA"

Each row is normalized to this project's minimal example schema: qid,
dataset, split, query, context, gold_answer (see gate_rlm/data.py and every
other scripts/prepare_*.py for the same shape).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

from gate_rlm.data import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/multidata_probe.jsonl"

REPOQA_URL = (
    "https://github.com/evalplus/repoqa_release/releases/download/"
    "2024-06-23/repoqa-2024-06-23.json.gz"
)


def load_babilong() -> dict:
    from datasets import load_dataset

    ex = load_dataset("RMT-team/babilong", "4k")["qa1"][0]
    return {
        "qid": "multidata-babilong-qa1-4k-0",
        "dataset": "babilong",
        "split": "test",
        "query": ex["question"],
        "context": ex["input"],
        "gold_answer": ex["target"],
    }


def load_browsecomp() -> dict:
    row = json.loads((ROOT / "data/processed/browsecomp.jsonl").read_text().splitlines()[0])
    return {
        "qid": "multidata-browsecomp-" + row["qid"],
        "dataset": "browsecomp_plus",
        "split": "test",
        "query": row["query"],
        "context": row["context"],
        "gold_answer": row["gold_answer"],
    }


def load_hotpotqa() -> dict:
    from datasets import load_dataset

    ex = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")[0]
    titles = ex["context"]["title"]
    sentences = ex["context"]["sentences"]
    context = "\n\n".join(
        f"{title}\n" + " ".join(sents) for title, sents in zip(titles, sentences)
    )
    return {
        "qid": "multidata-hotpotqa-" + ex["id"],
        "dataset": "hotpotqa",
        "split": "test",
        "query": ex["question"],
        "context": context,
        "gold_answer": ex["answer"],
    }


def load_oolong() -> dict:
    from datasets import load_dataset

    ex = load_dataset("oolongbench/oolong-real", "toy_dnd", split="test")[0]
    return {
        "qid": "multidata-oolong-0",
        "dataset": "oolong",
        "split": "test",
        "query": ex["question"],
        "context": ex["context_window_text"],
        "gold_answer": str(ex["answer"]),
    }


def load_repoqa() -> dict:
    with urllib.request.urlopen(REPOQA_URL, timeout=120) as resp:
        data = json.loads(gzip.decompress(resp.read()))
    entry = data["python"][0]
    needle = entry["needles"][0]
    context = "\n\n".join(f"# {path}\n{text}" for path, text in entry["content"].items())
    return {
        "qid": f"multidata-repoqa-{entry['repo'].replace('/', '_')}",
        "dataset": "repoqa",
        "split": "test",
        "query": needle["description"],
        "context": context,
        "gold_answer": needle["name"],
    }


def main() -> None:
    loaders = {
        "babilong": load_babilong,
        "browsecomp_plus": load_browsecomp,
        "hotpotqa": load_hotpotqa,
        "oolong": load_oolong,
        "repoqa": load_repoqa,
    }
    rows = []
    for name, loader in loaders.items():
        print(f"loading {name}...", flush=True)
        row = loader()
        row["context_tokens"] = len(row["context"]) // 4  # rough estimate, not tiktoken-exact
        rows.append(row)
        print(f"  qid={row['qid']}  context_chars={len(row['context'])}  gold={row['gold_answer']!r}")

    write_jsonl(rows, OUT)
    print(f"\nwrote {len(rows)} examples -> {OUT}")


if __name__ == "__main__":
    main()
