"""Build the BrowseComp-Plus subset (D2) with gold / evidence labels.

Step 1 (once): clone texttron/BrowseComp-Plus and decrypt the queries:
    python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl
Copy that file to data/raw/browsecomp_plus_decrypted.jsonl and the two qrels files
(topics-qrels/qrel_golds.txt, qrel_evidence.txt) to data/raw/.

Step 2:
    python scripts/prepare_browsecomp.py --n 100 --max-docs 50 --max-chars-per-doc 12000

Each query's context = its gold + evidence documents + sampled hard negatives,
shuffled with a fixed seed. All documents share one length cap, so length is not
a cue for relevance. The reduced corpus size is a deliberate cost decision;
report it in the paper.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from gate_rlm.data import assign_split, doc_fingerprints, read_jsonl, write_jsonl
from gate_rlm.router import estimate_tokens


def read_qrels(path: Path) -> dict:
    out = defaultdict(set)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3] != "0":
            out[parts[0]].add(parts[2])
    return out


def doc_list(row: dict, key: str) -> list:
    docs = row.get(key) or []
    return [{"docid": str(d.get("docid")), "text": d.get("text", "")} for d in docs if d]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decrypted", default="data/raw/browsecomp_plus_decrypted.jsonl")
    ap.add_argument("--qrel-gold", default="data/raw/qrel_golds.txt")
    ap.add_argument("--qrel-evidence", default="data/raw/qrel_evidence.txt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-docs", type=int, default=50)
    ap.add_argument("--max-chars-per-doc", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-fraction", type=float, default=0.3)
    ap.add_argument("--out", default="data/processed/browsecomp.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold_q = read_qrels(Path(args.qrel_gold))
    evid_q = read_qrels(Path(args.qrel_evidence))
    corpus = None  # loaded lazily only if the decrypted rows lack document text

    rows = list(read_jsonl(args.decrypted))
    rng.shuffle(rows)
    out = []
    for row in rows:
        if len(out) >= args.n:
            break
        qid = str(row["query_id"])
        docs = {d["docid"]: d for key in ("gold_docs", "evidence_docs", "negative_docs")
                for d in doc_list(row, key)}
        negatives = [d["docid"] for d in doc_list(row, "negative_docs")]
        if not docs:  # fall back to the public corpus + qrels
            if corpus is None:
                from datasets import load_dataset

                corpus = {str(d["docid"]): d for d in
                          load_dataset("Tevatron/browsecomp-plus-corpus", split="train")}
            ids = gold_q[qid] | evid_q[qid]
            docs = {i: {"docid": i, "text": corpus[i]["text"]} for i in ids if i in corpus}
        gold_ids = [i for i in gold_q[qid] if i in docs]
        evid_ids = [i for i in evid_q[qid] | gold_q[qid] if i in docs]
        if not gold_ids:
            continue
        rng.shuffle(negatives)
        chosen = evid_ids + [i for i in negatives if i not in evid_ids]
        chosen = chosen[: max(args.max_docs, len(evid_ids))]
        rng.shuffle(chosen)

        parts, capped = [], {}
        for n, docid in enumerate(chosen):
            text = docs[docid]["text"][: args.max_chars_per_doc]
            capped[docid] = text
            parts.append(f"### Document {n + 1}\n{text}")
        context = "\n\n".join(parts)
        out.append({
            "qid": f"browsecomp-{qid}",
            "dataset": "browsecomp_plus",
            "split": assign_split(f"browsecomp-{qid}", args.val_fraction),
            "query": row["query"],
            "context": context,
            "gold_answer": row["answer"],
            "gold_fingerprints": [doc_fingerprints(capped[i]) for i in gold_ids],
            "gold_mode": "any",   # a gold doc alone suffices to derive the answer
            "evidence_fingerprints": [doc_fingerprints(capped[i]) for i in evid_ids],
            "length_bucket": "long",
            "complexity": "complex",
            "context_tokens": estimate_tokens(context),
            "meta": {"n_docs": len(chosen), "n_gold": len(gold_ids), "n_evidence": len(evid_ids)},
        })
    print(f"wrote {write_jsonl(out, args.out)} examples to {args.out}")


if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    main()
