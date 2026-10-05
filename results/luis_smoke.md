# Smoke test: all stopping signals on 6 long-context benchmarks (smoke5)

Run 2026-10-01 17:09–19:17 WIB on jupyter-04 (Mac Studio M3 Ultra), 60 questions, 0 server crashes.
Run log: `docs/run_notes_smoke5.md`. Code: `scripts/prep_smoke5.py`, `scripts/run_smoke5.py`, `scripts/score_smoke5.py`,
`scripts/bcp_rlm_runner.py`. Raw data on server: `work/smoke5run/smoke5_checkpoint.zip` (11.4 MB),
`smoke5_scores.csv`, `smoke5_per_turn.csv`.

## Setup
| item | value |
|---|---|
| RLM | rlms 0.1.3, root Qwen3.5-35B-A3B-4bit, worker Qwen3.5-2B-bf16 (mlx_lm, fresh-key sampler) |
| Trajectory | plain RLM (no stop rule, no hints); 3 parallel lanes |
| Budget | **10 turns, 10-min safety stop** (incl. probe time) — shorter than all150 (20 turns / 30 min) |
| Probe (every turn >= 2, hidden) | greedy draft + token probs + top-5 logprobs (FLARE min_p, DRAGIN-style), P(True), 10 samples T=1.0 clustered by DeBERTa-large-MNLI (SE, SE_w), per-turn doc ids (Knee/Chao) |
| Judge | root model, BrowseComp grader prompt; every distinct draft judged too |
| Sampling seed | 20261002 |

## Datasets (10 each)
| set | source | context | answer |
|---|---|---|---|
| hotpotqa | hotpot_qa distractor validation | 2.6–10k chars, 10 paragraphs | short span |
| babilong | RMT-team/babilong 64k, qa1–qa5 (2 each) | 234–269k chars | one word |
| codeqa | LongBench-v2 Code Repository Understanding (short+medium) | 101–635k chars | A–D |
| bcp | BrowseComp-Plus, 10 ids unused by all earlier runs | 0.8–4.4M chars | short string |
| oolong | oolong-synth validation trec_coarse, 32k tokens, one window | 76.8k chars (787 records) | label / count / comparison |
| oolong_pairs | same window, 10 of the 20 RLM-paper pair tasks (App. E.1); GT computed from labels | 76.8k chars | list of 209–8001 pairs, pair F1 |

## Accuracy and cost
| set | judge correct | task score | mean wall s | mean turns | 10-min stops | mean probe s (share of wall) |
|---|---|---|---|---|---|---|
| hotpotqa | **9/10** | 0.90 | 75 | 4.1 | 0 | 32 (43%) |
| babilong | **6/10** | 0.60 | 269 | 7.6 | 1 | 71 (26%) |
| codeqa | **6/10** | 0.60 | 395 | 8.2 | 2 | 92 (23%) |
| bcp | **5/10** | 0.50 | 404 | 8.7 | 4 | 88 (22%) |
| oolong | **1/9** (+1 unparsed) | 0.10 | 347 | 8.1 | 2 | 70 (20%) |
| oolong_pairs | 0/9 (judge not meaningful) | **pair F1 0.05** | 529 | 4.8 | 7 | 58 (11%) |
| **all** | **27/58** judged | | 337 | 6.9 | 16/60 | 69 |

BABILong by task: qa1 2/2, qa2 0/2, qa3 0/2, qa4 2/2, qa5 2/2.

## Do the signals separate right from wrong drafts? (probe-level AUROC)
Non-UNKNOWN drafts, each judged correct/wrong; all turns pooled (turns of one question are not independent).
| set | probes | judged drafts (correct) | FLARE min_p | P(True) | −SE | −SE_w |
|---|---|---|---|---|---|---|
| babilong | 69 | 31 (13) | .88 | **.98** | .97 | .97 |
| bcp | 83 | 30 (21) | **.91** | .77 | .80 | .83 |
| codeqa | 75 | 38 (24) | .67 | .76 | .68 | .69 |
| hotpotqa | 31 | 23 (15) | .59 | .78 | .42 | .42 |
| oolong | 74 | 20 (2) | .53 | 1.0 (2 pos) | .78 | .67 |
| oolong_pairs | 45 | 4 (0) | – | – | – | – |
| **all** | 377 | 146 (75) | .77 | **.83** | .76 | .74 |

## Stopping opportunities
| case | count | questions |
|---|---|---|
| Final wrong, but a correct draft appeared earlier ("found but didn't stop") | ~5/60 (bcp 1, codeqa 2, oolong 2; quick count from per-turn CSV) | e.g. bcp 1041 (Adaku, min_p 1.0, SE 0), codeqa 66f2e874 (C, SE 0), oolong 15000207 |
| Final right without any correct draft (draft probe names a bridge entity or UNKNOWN) | 3/60 | hotpotqa 5ab294a6 (Jenson Button), hotpotqa 5a7de63c (Frank Borman), oolong 15000213 |
| Confident-wrong drafts (min_p 1.0 and SE 0 on a wrong answer) | several | oolong 15000221 (abbreviation), babilong qa1-86 t5 (Paris) |

## Takeaways
1. The pipeline runs end to end on all 6 benchmarks; every signal is recorded (SE varies, so the fresh-key sampler works).
2. Signals discriminate well where answers are short entities (BABILong, BCP: AUROC .8–.98), weaker on multiple-choice
   code (.67–.76) and on HotpotQA, where most drafts are already right (SE 0 nearly always).
3. P(True) is the best single signal overall (.83); FLARE min_p is best on BCP (.91).
4. OOLONG-Pairs is not usable with the 48-token draft probe: drafts are UNKNOWN or truncated lists, 7/10 hit the
   10-min stop. Needs a list-aware probe (e.g. ask for the pair count, or score partial lists) or exclusion.
5. OOLONG aggregation is hard for this setup (1/9): the root writes counting code but labels the 787 records badly.
6. The 10-min budget binds on BCP and OOLONG-Pairs (11 of 16 stops). Full runs should use 20 turns / 30 min.
7. Probe overhead 11–43% of wall time (highest on HotpotQA, where runs are short).

## Caveats
Smoke scale (10 per set); thresholds not tuned; judge is the same root model; OOLONG at 32k (RLM paper used 131k);
CodeQA letter regex under-counts (judge used instead); answer parser sometimes grabs code lines (judge reads the full response);
Knee/Chao doc ids exist for all sets but only hotpotqa and bcp have gold documents.

## Per question
Last probe = `turn:draft/min_p/P(True)/SE` of the final probe (empty = all drafts UNKNOWN). Judge Y/N/- (- = verdict not parsed).
| set | Q | wall s | turns | 10-min stop | judge | score | final answer (parsed) | ground truth | last probe |
|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 5a75fa14 | 81 | 4 |  | N | 0 | (Explanation line) | 2018 Unibet Premier League | 4:Premier League Darts/1.00/0.98/0.00 |
| hotpotqa | 5ae46c9d | 106 | 5 |  | Y | 1 | Nobel Prize in Physics | Nobel Prize | 5:Nobel Prize in Physics/1.00/0.99/0.00 |
| hotpotqa | 5a8e08d3 | 16 | 2 |  | Y | 1 | Bayern Munich | Bayern Munich | 2:Bayern Munich/1.00/0.98/0.00 |
| hotpotqa | 5a886364 | 30 | 2 |  | Y | 1 | (Explanation line) | American musical duo The Books | 2:The Books/0.78/0.97/0.00 |
| hotpotqa | 5ab294a6 | 35 | 2 |  | Y | 1 | The son of John Button (...) | McLaren-Honda | 2:Jenson Button/0.88/0.88/0.33 |
| hotpotqa | 5a7b88a8 | 130 | 7 |  | Y | 1 | 1618 | 1618 | 7:1618/1.00/0.98/0.00 |
| hotpotqa | 5ab67d20 | 73 | 5 |  | Y | 1 | Labrador Retriever | Labrador Retriever | 5:Labrador Retriever/1.00/0.99/0.00 |
| hotpotqa | 5a8b24f2 | 22 | 2 |  | Y | 1 | Albert Heijn founded Ahold | Koninklijke Ahold N.V. | 2:Ahold/0.69/0.56/0.33 |
| hotpotqa | 5a7de63c | 76 | 4 |  | Y | 1 | 1928 | 1928 | 4:Frank Borman/0.78/0.62/0.00 |
| hotpotqa | 5a7b1023 | 179 | 8 |  | Y | 1 | Ride a Wild Pony | Ride a Wild Pony | 8:Ride a Wild Pony/1.00/0.99/0.00 |
| babilong | qa1-20 | 181 | 7 |  | Y | 1 | hallway | hallway | 7:Hallway/0.88/0.95/0.00 |
| babilong | qa1-86 | 226 | 7 |  | Y | 1 | Garden | garden | 7:Garden/1.00/0.50/0.00 |
| babilong | qa2-18 | 439 | 10 |  | N | 0 | Churn | garden | 10:In the can/0.47/0.32/0.90 |
| babilong | qa2-69 | 600 | 8 | yes | N | 0 | (code line) | bathroom | 9:In her hand/0.25/0.02/1.75 |
| babilong | qa3-47 | 283 | 10 |  | N | 0 | (code line) | office | |
| babilong | qa3-51 | 373 | 10 |  | N | 0 | Valley | bedroom | |
| babilong | qa4-89 | 119 | 6 |  | Y | 1 | bathroom | bathroom | 6:bathroom/0.78/0.99/0.00 |
| babilong | qa4-45 | 290 | 8 |  | Y | 1 | office | office | 8:office/0.69/0.59/0.00 |
| babilong | qa5-11 | 127 | 6 |  | Y | 1 | Jeff | Jeff | 6:Jeff/1.00/0.90/0.00 |
| babilong | qa5-83 | 56 | 4 |  | Y | 1 | milk | milk | 4:milk/0.88/1.00/0.00 |
| codeqa | 66f3ad93 | 227 | 8 |  | Y | 1 | C | C | 8:C/1.00/0.99/0.00 |
| codeqa | 66f530ce | 388 | 8 |  | Y | 1 | C | C | 8:Option C/0.61/0.75/0.00 |
| codeqa | 66f2e874 | 600 | 8 | yes | N | 0 | (code line) | C | 8:C/0.78/0.44/0.00 |
| codeqa | 6708a096 | 505 | 9 |  | Y | 0* | (sentence, Option D) | D | 9:D/0.88/0.91/0.00 |
| codeqa | 66fa7269 | 600 | 5 | yes | N | 0 | (code line) | B | 6:B/0.25/0.10/0.33 |
| codeqa | 66ec3644 | 363 | 9 |  | N | 0 | Option (C) | D | 9:C/1.00/0.94/0.00 |
| codeqa | 66fa700b | 308 | 10 |  | Y | 1 | C | C | 11:C/0.69/0.22/0.50 |
| codeqa | 66ebd3ba | 215 | 5 |  | N | 0 | (none) | A | 5:C/0.88/0.92/0.00 |
| codeqa | 66ed3e90 | 511 | 10 |  | Y | 1 | C | C | 10:C/0.54/0.98/0.00 |
| codeqa | 66fcf36f | 227 | 10 |  | Y | 1 | D | D | 11:D/1.00/0.95/0.00 |
| bcp | 784 | 600 | 10 | yes | N | 0 | (code line) | Jacqueline Georgette Cant | 10:Maurice Chevalier/0.54/0.00/0.50 |
| bcp | 237 | 600 | 10 | yes | N | 0 | (code line) | Edward Winslow | |
| bcp | 1012 | 116 | 5 |  | Y | 1 | 22, September, 2023 | 22 September 2023 | 5:September 22, 2023/0.78/0.84/0.00 |
| bcp | 1041 | 601 | 7 | yes | N | 0 | (code line) | Adaku | 8:Adaku/1.00/0.03/0.00 |
| bcp | 530 | 185 | 9 |  | Y | 1 | Tuesday, January 14 | Tuesday, January 14 | 9:Tuesday, January 14/0.88/0.97/0.50 |
| bcp | 560 | 564 | 10 |  | N | 0 | (code line) | David Leonard Landau | 10:Terrance Stanley F.../0.54/0.03 |
| bcp | 618 | 601 | 10 | yes | N | 0 | (code line) | FORMULA 1 HEINEKEN GRANDE ... | 11:Sakhir Grand Prix/0.78/0.12 |
| bcp | 1260 | 214 | 6 |  | Y | 1 | Aynabaji | Aynabaji | 6:Aynabaji/1.00/0.94/0.00 |
| bcp | 202 | 382 | 10 |  | Y | 1 | Gloria-Sophie Burkandt | Gloria-Sophie Burkandt | 11:Gloria-Sophie Burkandt/1.00/0.56/0.00 |
| bcp | 126 | 172 | 10 |  | Y | 1 | New directions in the treatm... | New directions in the tre... | 10:New directions in.../1.00/0.35 |
| oolong | 15000201 | 457 | 8 |  | N | 0 | human being | entity | 4:abbreviation/0.61/0.15/0.00 |
| oolong | 15000200 | 236 | 10 |  | N | 0 | Label: abstract concept | abbreviation | |
| oolong | 15000218 | 530 | 10 |  | N | 0 | (code line) | 123 | 11:66/0.88/0.15/0.00 |
| oolong | 15000207 | 600 | 7 | yes | N | 0 | (code line) | more common than | 8:numeric value is more.../0.88/0.68/0.00 |
| oolong | 15000221 | 601 | 7 | yes | N | 0 | (code line) | description and abstract concept | 8:abbreviation/1.00/0.22/0.00 |
| oolong | 15000220 | 245 | 10 |  | N | 0 | Label: None | numeric value | 10:human being/0.25/0.01/0.33 |
| oolong | 15000214 | 472 | 9 |  | N | 0 | 194 | 127 | 9:194/0.69/0.15/1.36 |
| oolong | 15000213 | 118 | 5 |  | Y | 1 | (Explanation line) | more common than | 3:same frequency as/0.42/0.29/0.33 |
| oolong | 15000219 | 111 | 8 |  | N | 0 | 1 | 73 | |
| oolong | 15000208 | 104 | 7 |  | - | 0 | numeric value is less common | more common than | 7:entity is more common.../0.32/0.13 |
| oolong_pairs | t1 | 278 | 5 |  | N | F1 .037 | pair list (12455, 13578) ... | 8001 pairs | 5:(12455, 13578) .../0.14/0.13/1.40 |
| oolong_pairs | t4 | 600 | 4 | yes | N | 0 | (code line) | 3160 pairs | |
| oolong_pairs | t5 | 443 | 10 |  | N | 0 | (code line) | 2485 pairs | |
| oolong_pairs | t7 | 601 | 4 | yes | N | 0 | (code line) | 4656 pairs | |
| oolong_pairs | t9 | 601 | 3 | yes | N | 0 | (code line) | 1830 pairs | |
| oolong_pairs | t12 | 368 | 4 |  | - | F1 .471 | pair list (15245, 19125) ... | 1505 pairs | |
| oolong_pairs | t15 | 601 | 5 | yes | N | 0 | (code line) | 1112 pairs | 5:(16303, 10352)/0.47/0.05/0.90 |
| oolong_pairs | t16 | 600 | 4 | yes | N | 0 | (code line) | 209 pairs | |
| oolong_pairs | t18 | 601 | 5 | yes | N | 0 | (code line) | 980 pairs | 5:(13973, 16303)/0.37/0.00/1.03 |
| oolong_pairs | t20 | 600 | 4 | yes | N | 0 | (code line) | 314 pairs | |

\* codeqa 6708a096: judge Y, letter regex missed "Option D" in a sentence.
