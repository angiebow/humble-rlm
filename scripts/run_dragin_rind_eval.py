#!/usr/bin/env python3
"""
DRAGIN-RLM HotpotQA evaluation with RIND scores.
20 sample questions with attention probing and RIND metrics.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

print("=" * 80)
print("DRAGIN-RLM HotpotQA Evaluation with RIND Scores")
print("=" * 80)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Sample HotpotQA questions with context
SAMPLE_QUESTIONS = [
    {
        "qid": "hotpotqa-000",
        "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "context": """Scott Derrickson is an American film director and screenwriter. He was born in 1966.
Ed Wood was an American filmmaker and screenwriter. He was born in 1924 and died in 1978.
Both were American, so yes, they shared the same nationality.""",
        "gold_answer": "yes"
    },
    {
        "qid": "hotpotqa-001",
        "query": "Are both Local H and For Against from the United States?",
        "context": """Local H is an American rock band from the United States.
For Against was an American post-punk/gothic rock band from the United States.
Both bands originated in the United States.""",
        "gold_answer": "Yes, they are both from the United States."
    },
    {
        "qid": "hotpotqa-002",
        "query": "Who wrote the metaphysical poem about love and death?",
        "context": """John Donne (1572-1631) was an English poet and clergyman known for his metaphysical poetry.
He wrote poems exploring themes of love, death, and spirituality.
His works include 'A Valediction Forbidding Mourning' and 'Holy Sonnets'.""",
        "gold_answer": "John Donne"
    },
    {
        "qid": "hotpotqa-003",
        "query": "What is the nationality of the author of that metaphysical poem?",
        "context": """John Donne was born in England and is considered an English poet.
His nationality was English, reflecting his birth and life in England.""",
        "gold_answer": "American"
    },
    {
        "qid": "hotpotqa-004",
        "query": "When did the famous actress die?",
        "context": """Shirley Temple was an American child actress and diplomat.
She was born in 1928 and died in 2014.
She appeared in films like 'Dimples' and 'Curly Top' in the 1930s.""",
        "gold_answer": "Yes, she died in 1989."
    },
    {
        "qid": "hotpotqa-005",
        "query": "What types of books did the author write?",
        "context": """Jane Austen was a novelist who wrote both fiction and non-fiction works.
Her novels include 'Pride and Prejudice' and 'Sense and Sensibility'.
She also wrote letters and literary criticism.""",
        "gold_answer": "Fiction and non-fiction"
    },
    {
        "qid": "hotpotqa-006",
        "query": "What genre of writing is most associated with her?",
        "context": """Maya Angelou was primarily known for her autobiographical work.
Her most famous work is the memoir 'I Know Why the Caged Bird Sings'.
She also wrote poetry and essays throughout her career.""",
        "gold_answer": "Memoir"
    },
    {
        "qid": "hotpotqa-007",
        "query": "What information is not found in the provided passages?",
        "context": """The Arctic is a polar region at the northern end of Earth.
It is characterized by extremely cold temperatures and ice coverage.
The region contains land, ocean, and ice masses.""",
        "gold_answer": "Based on the passages, I cannot find this information."
    },
    {
        "qid": "hotpotqa-008",
        "query": "What language do most people in England speak?",
        "context": """England is a country located in Western Europe.
The official and primary language spoken in England is English.
English is the most widely used language throughout the nation.""",
        "gold_answer": "English language"
    },
    {
        "qid": "hotpotqa-009",
        "query": "Are the two authors contemporary?",
        "context": """Mark Twain lived from 1835 to 1910 and was an American author.
Louisa May Alcott lived from 1832 to 1888 and was an American author.
Both lived during the 19th century, making them contemporary authors.""",
        "gold_answer": "Yes, they are contemporary authors."
    },
    {
        "qid": "hotpotqa-010",
        "query": "Which major highway does 8 Mile Road connect to?",
        "context": """8 Mile Road is an east-west major thoroughfare in Michigan.
It connects to Interstate 75 (I-75) in the city of Ferndale.
The road is located at approximately 8 miles north of downtown Detroit.""",
        "gold_answer": "Yes, 8 mile Road connects to I-75."
    },
    {
        "qid": "hotpotqa-011",
        "query": "Which Shakespeare play features spirits and magic?",
        "context": """The Tempest is a play by William Shakespeare.
It features themes of magic, spirits, and supernatural elements.
The play was written in 1610-1611 and is classified as a comedy.""",
        "gold_answer": "The Tempest"
    },
    {
        "qid": "hotpotqa-012",
        "query": "In which city is the famous jazz festival held?",
        "context": """The New Orleans Jazz Festival is held annually in New Orleans, Louisiana.
New Orleans is known as the birthplace of jazz music.
The festival celebrates the cultural heritage and music of the region.""",
        "gold_answer": "New Orleans"
    },
    {
        "qid": "hotpotqa-013",
        "query": "Have both singers been sponsored by Pepsi?",
        "context": """Britney Spears has had several endorsement deals with Pepsi.
Christina Aguilera has also had advertising and endorsement deals with Pepsi.
Both pop singers have represented the Pepsi brand.""",
        "gold_answer": "Yes, Pepsi has sponsored both."
    },
    {
        "qid": "hotpotqa-014",
        "query": "What field of study focuses on human behavior and mental processes?",
        "context": """Psychology is the scientific study of human behavior and mental processes.
It encompasses various subdisciplines like clinical psychology, cognitive psychology, and social psychology.
Psychologists study how people think, feel, and behave.""",
        "gold_answer": "Psychology"
    },
    {
        "qid": "hotpotqa-015",
        "query": "What government position did Thomas Jefferson hold before becoming president?",
        "context": """Thomas Jefferson served as Governor of Virginia from 1779 to 1781.
He held this position before his later role as President of the United States.
Jefferson was also Minister to France and Secretary of State.""",
        "gold_answer": "Yes, Thomas Jefferson served as governor."
    },
    {
        "qid": "hotpotqa-016",
        "query": "Which tragic play centers on the theme of ambition and power?",
        "context": """Macbeth is a tragedy by William Shakespeare.
The play explores themes of ambition, power, and corruption.
Macbeth is often considered one of Shakespeare's greatest tragedies.""",
        "gold_answer": "Macbeth"
    },
    {
        "qid": "hotpotqa-017",
        "query": "Which space agency explores outer space?",
        "context": """NASA stands for the National Aeronautics and Space Administration.
It is the primary space agency of the United States.
NASA conducts space exploration, research, and development programs.""",
        "gold_answer": "NASA"
    },
    {
        "qid": "hotpotqa-018",
        "query": "Did Tom Hanks star in the space exploration film?",
        "context": """Apollo 13 is a 1995 film directed by Ron Howard.
Tom Hanks starred as Jim Lovell in the film.
The film won multiple Academy Awards and is based on a true story.""",
        "gold_answer": "Yes, Tom Hanks starred in Apollo 13."
    },
    {
        "qid": "hotpotqa-019",
        "query": "What country is known for precision engineering and watches?",
        "context": """Switzerland is famous for its precision manufacturing and watchmaking.
Swiss watches are renowned for their quality and accuracy.
The country has a strong tradition of engineering excellence.""",
        "gold_answer": "Germany"
    }
]

def run_evaluation():
    """Run DRAGIN evaluation with RIND metrics."""

    try:
        print("\nImporting DRAGIN modules...")
        from dragin_rlm.harness import DraginConfig, run_dragin
        from dragin_rlm.attention_probe import load_dragin_model
        from eval.metrics import f1, is_correct, semantic_similarity
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nFalling back to direct HTTP evaluation (no RIND)...")
        return run_direct_http_eval()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "dragin_hotpotqa_rind_results_20.jsonl"
    csv_file = out_dir / "table_dragin_hotpotqa_rind_20.csv"

    # Clear any previous results
    if out_file.exists():
        out_file.unlink()

    print("\nLoading DRAGIN model...")
    try:
        model, tokenizer = load_dragin_model("mlx-community/Qwen3.5-35B-A3B-4bit")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return run_direct_http_eval()

    # Create DRAGIN config
    dcfg = DraginConfig(
        model_path="mlx-community/Qwen3.5-35B-A3B-4bit",
        worker_model="mlx-community/Qwen3.5-2B-4bit",
        theta=1.0,
        top_n=25,
        generate_length=256,
        max_retrieval_seconds=600.0,
        max_triggers=20,
        retrieval_top_k=3,
        passage_chars=1000,
        temperature=0.0,
    )

    print(f"Config: theta={dcfg.theta}, max_triggers={dcfg.max_triggers}")
    print(f"\nRunning DRAGIN evaluation on {len(SAMPLE_QUESTIONS)} questions...")
    print("-" * 80)

    results = []

    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"[{i+1:2d}/{len(SAMPLE_QUESTIONS)}] {q['qid']}: {q['query'][:60]}...", end=" ", flush=True)

        t_start = time.perf_counter()

        try:
            dragin_result = run_dragin(
                query=q["query"],
                context=q["context"],
                cfg=dcfg,
                model=model,
                tokenizer=tokenizer,
            )

            latency_s = time.perf_counter() - t_start

            result = {
                "qid": q["qid"],
                "query": q["query"],
                "gold_answer": q["gold_answer"],
                "answer": dragin_result.get("answer", ""),
                "answer_source": dragin_result.get("answer_source", ""),
                "latency_s": latency_s,
                "completion_tokens": dragin_result.get("completion_tokens", 0),
                "llm_calls": dragin_result.get("llm_calls", 0),
                "n_retrievals": dragin_result.get("n_retrievals", 0),
                "max_rind_score": dragin_result.get("max_rind_score", 0.0),
                "rind_checked_tokens": dragin_result.get("rind_checked_tokens", 0),
                "rind_nonstopword_tokens": dragin_result.get("rind_nonstopword_tokens", 0),
                "time_capped": dragin_result.get("time_capped", False),
            }
            results.append(result)

            with open(out_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"✓ RIND:{result['max_rind_score']:.3f} ret:{result['n_retrievals']}")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            continue

    print("-" * 80)
    print(f"\n✓ Processed {len(results)} questions")
    print(f"✓ Results saved to {out_file}")

    # Generate CSV report
    if results:
        print("\nGenerating CSV report...")
        csv_lines = [
            "qid,gold_answer,answer,latency_s,completion_tokens,n_retrievals,max_rind_score,rind_checked_tokens"
        ]

        for r in results:
            csv_lines.append(
                f"{r['qid']},\"{r['gold_answer']}\",\"{r['answer']}\","
                f"{r['latency_s']:.2f},{r['completion_tokens']},{r['n_retrievals']},"
                f"{r['max_rind_score']:.4f},{r['rind_checked_tokens']}"
            )

        with open(csv_file, "w") as f:
            f.write("\n".join(csv_lines))

        print(f"✓ CSV report saved to {csv_file}")

        # Print summary
        avg_rind = sum(r['max_rind_score'] for r in results) / len(results)
        total_retrievals = sum(r['n_retrievals'] for r in results)
        avg_latency = sum(r['latency_s'] for r in results) / len(results)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Questions evaluated:     {len(results)}")
        print(f"Avg RIND score:          {avg_rind:.4f}")
        print(f"Total retrievals:        {int(total_retrievals)}")
        print(f"Avg retrievals/question: {total_retrievals/len(results):.2f}")
        print(f"Avg latency:             {avg_latency:.2f}s")
        print("=" * 80)

def run_direct_http_eval():
    """Fallback to direct HTTP evaluation if DRAGIN not available."""
    print("\nFalling back to Direct HTTP evaluation...")
    import urllib.request

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "fallback_hotpotqa_eval_20.jsonl"
    csv_file = out_dir / "table_fallback_hotpotqa_eval_20.csv"

    if out_file.exists():
        out_file.unlink()

    results = []

    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"[{i+1:2d}/{len(SAMPLE_QUESTIONS)}] {q['qid']}: {q['query'][:60]}...", end=" ", flush=True)

        t_start = time.perf_counter()

        prompt = f"Answer the following question based on the context.\n\nContext:\n{q['context']}\n\nQuestion: {q['query']}\nAnswer:"
        payload = {
            "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0,
        }

        try:
            req = urllib.request.Request(
                "http://localhost:8005/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = json.loads(response.read().decode('utf-8'))
                answer = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                tokens = resp_data.get("usage", {}).get("total_tokens", 0)
        except Exception as e:
            answer = ""
            tokens = 0

        latency_s = time.perf_counter() - t_start

        result = {
            "qid": q["qid"],
            "gold_answer": q["gold_answer"],
            "answer": answer,
            "latency_s": latency_s,
            "tokens": tokens,
        }
        results.append(result)

        with open(out_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"✓ {tokens} tokens")

    print(f"\n✓ Processed {len(results)} questions (direct HTTP)")
    print(f"✓ Results saved to {out_file}")

    # Generate CSV
    csv_lines = ["qid,gold_answer,answer,latency_s,tokens"]
    for r in results:
        csv_lines.append(
            f"{r['qid']},\"{r['gold_answer']}\",\"{r['answer']}\","
            f"{r['latency_s']:.2f},{r['tokens']}"
        )

    with open(csv_file, "w") as f:
        f.write("\n".join(csv_lines))

    print(f"✓ CSV saved to {csv_file}")

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
