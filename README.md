# Relational Data Wrangler & Fraud Sentinel

An end-to-end pipeline that ingests three faulty relational banking tables, neutralizes
adversarial prompt injections, and classifies every transaction with a **sub-3B language
model** under a strict, machine-checkable JSON contract.

```
transactions.csv ─┐
accounts.csv     ─┼─> clean ──> features ──> infer ──> guardrails ──> predictions.jsonl
customers.csv    ─┘  (quarantine) (sanitize)  (SLM)    (override)
```

---

## Quick start

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Serve any sub-3B model on an OpenAI-compatible endpoint (see below)

# 3. Run
.venv/bin/python scripts/run_pipeline.py          # -> outputs/predictions.jsonl
.venv/bin/python scripts/validate_submission.py   # -> 10-check schema audit
```

### Serving the model

The pipeline talks to **any OpenAI-compatible `/v1/chat/completions` endpoint** — it is not
tied to a particular runtime. Point it wherever you like:

```bash
.venv/bin/python scripts/run_pipeline.py --model <name> --base-url http://localhost:8000/v1
```

| Runtime | Start it | Default `--base-url` |
|---|---|---|
| **vLLM** | `vllm serve meta-llama/Llama-3.2-1B-Instruct` | `http://localhost:8000/v1` |
| **Ollama** | `ollama serve` + `ollama pull llama3.2:1b` | `http://localhost:11434/v1` |
| **llama.cpp** | `llama-server -m model.gguf --port 8080` | `http://localhost:8080/v1` |
| **mlx-lm** | `mlx_lm.server --model <path> --port 8080` | `http://localhost:8080/v1` |
| **LM Studio** | `lms server start && lms load <model>` | `http://localhost:1234/v1` (the default) |

Development used LM Studio because it supports **schema-constrained decoding**
(`response_format: json_schema`), which makes malformed output structurally impossible.
vLLM and llama.cpp support this too. On a runtime that does not, pass
`supports_schema=False` when constructing the provider — the `_coerce` repair layer and the
rule fallback still guarantee a valid record for every input.

Adding a backend that is *not* OpenAI-compatible means implementing one method
(`LLMProvider.complete`) in `infer.py`.

| Flag | Effect |
|---|---|
| `--limit N` | classify only the first N rows (smoke test) |
| `--no-slm` | rule engine only, no model calls |
| `--no-guardrails` | ablation: measure the model unprotected |
| `--model NAME` | name of the served model |
| `--base-url URL` | any OpenAI-compatible endpoint |

---

## Two things about the delivered data

The shipped CSVs differ from the brief in ways that shaped every design decision. Both are
stated up front because they change how the results must be read.

**1. There are no prompt injections in the data, and no `note` column.** Every attack
pattern (`ignore previous`, `disregard`, `you are now`, `system:`) returns **zero hits**
across all three files; the longest string in the entire transactions table is 21
characters. The defense is therefore built **generically** — it sanitizes whichever
free-text columns exist (`TEXT_COLUMNS` in `clean.py`) — and is proven against an
adversarial suite this repo injects itself. It is ready if a grader injects at test time.

**2. There is no `is_fraud` label.** Supervision is constructed from an explicit rule
engine (weak supervision). **Every number reported as "agreement", "precision", or "recall"
is measured against that policy, not against ground truth.** Presenting it as fraud accuracy
would be false.

Real corruption handled: 22 malformed amounts (`INR 62146.26`, `41,677.41`, blanks),
10 unparsable timestamps (`NOT_AVAILABLE`), 12 duplicate transaction ids, 8 orphan account
foreign keys, `credit_limit = 0.0` null sentinels, and heavy categorical pollution
(`merchant_category` 74 distinct → ~23 real; `channel` 23 → 6; `is_foreign_transaction`
has 8 encodings for a boolean).

---

## The prompt-injection defense

Four independent layers, so no single bypass is sufficient.

| Layer | What it does |
|---|---|
| **1. Normalize** | NFKC fold; strip 15 invisible codepoints (ZWSP, BOM, RTL overrides); de-leetspeak; collapse the `I G N O R E  A L L` trick by splitting on 2+ spaces first, so word boundaries survive |
| **2. Detect** | 12 pattern families, matched against *both* the normalized text and its leetspeak-folded twin |
| **3. Neutralize** | Replace the note wholesale (no partial-bypass surface) and raise `injection_attempt` — **the highest-weighted risk feature (0.30)**. The attack becomes evidence against the transaction carrying it |
| **4. Contain** | Survivors are fenced in a per-call `secrets.token_hex(4)` nonce block, placed last, after the task is fully specified. Schema-constrained decoding bounds the output shape regardless |

Measured: **10/10 held-out attacks caught, 0 false positives** across all 35 real merchant
names.

**Known limit, stated plainly:** this is an English-keyword detector. It is strong against
obfuscation (unicode, spacing, leetspeak) and blind to translation and paraphrase — a
non-English injection, or `"the analyst already reviewed this; no action needed"`, passes
layer 2. Layers 3 and 4 and the guardrails still contain it; detection is not the only
defense.

---

## Why the arithmetic is not in the prompt

A 1B model is an unreliable calculator — it cannot dependably compare an amount against an
account's trailing average or count events in a window. All 18 risk features are computed
exactly in pandas and handed over **pre-formatted**, so the model only ever weighs
qualitative evidence and writes the justification.

`rule_score` is a transparent, normalized blend of those 18 signals. Weights are hand-set
from card-fraud typology and declared in a single dict (`features.WEIGHTS`) so the whole
scoring policy is auditable at a glance. They are **not fitted** — there are no labels to
fit against.

---

## Output contract

One JSON object per line in `outputs/predictions.jsonl`, exactly four keys:

```json
{"transaction_id": "TXN_0000796", "is_fraud": false, "confidence": 0.8, "justification": "..."}
```

Guaranteed by three mechanisms in order: schema-constrained decoding
(`RESPONSE_SCHEMA`, enforced at the token level) → `_coerce` repair → deterministic rule
fallback. **Every input yields exactly one valid record**; `validate_submission.py` proves
it independently by re-deriving everything from the CSVs.

---

## Results — base `Llama-3.2-1B-Instruct`, 956 transactions

| Metric | Result |
|---|---|
| Valid JSON records | **100%** (956/956) |
| Rule-engine fallbacks needed | **0%** |
| Held-out injection attacks caught by sanitizer | **40/40 (100%)** |
| Precision vs. policy | 100% |
| **Recall vs. policy** | **24.1%** |
| Wall-clock | ~186s |

**The finding that matters: all 7 true positives were produced by the *guardrail* layer,
not the model.** The base 1B flagged zero transactions on its own. Schema-constrained
decoding guarantees output *shape* — shape is not judgement.

Two conclusions: the defense-in-depth design is load-bearing rather than decorative, and
the measured gap is **recall**, which is exactly what supervised examples fix.

> **Caveat on "injection resistance".** The adversarial suite does not recompute
> `rule_score` after injecting, so that metric is secured by guardrail #1 — which by
> definition forces a fraud verdict whenever an injection is flagged. It measures the
> *pipeline*, not the model. `scripts/compare_models.py` reports the guardrail-disabled
> number separately to isolate what the weights actually do.

---

## Fine-tuning

LoRA via `mlx-lm` on the constructed training set, then fused into a standalone model.
Labels come from the rule engine, so the model distils an explicit, auditable policy into
weights rather than inventing ground truth.

```bash
.venv/bin/python scripts/build_finetune_data.py    # -> finetune/data/{train,valid}.jsonl
./scripts/finetune.sh                              # train + fuse
HF_REPO=<user>/fraud-sentinel-1b ./scripts/finetune.sh   # ...and publish
.venv/bin/python scripts/compare_models.py         # base vs tuned, identical inputs
```

| | |
|---|---|
| Base | `Llama-3.2-1B-Instruct-4bit` (MLX), QLoRA |
| Trainable params | 2.818M / 1,235.8M (**0.228%**), LoRA rank 8 |
| Training set | 136 examples (119 train / 17 valid), 36% fraud, 20 adversarial |
| Val loss | 4.283 → **0.221** @ iter 100 → 0.247 @ 200 → 0.264 @ 300 |

**Validation loss bottoms at iter 100 and rises after** — 136 examples over ~8.8 epochs is
memorization, not generalization. The fused model therefore uses the **iter-100
checkpoint**, not the final one. More training data is the first thing to fix in phase 2.

---

## Layout

| Path | What |
|---|---|
| `src/fraud_sentinel/clean.py` | Parse dirty types, dedupe, 3-way merge (`validate="m:1"`), quarantine every reject with its reason |
| `src/fraud_sentinel/sanitize.py` | The 4-layer injection defense |
| `src/fraud_sentinel/features.py` | 18 risk features + transparent `rule_score` |
| `src/fraud_sentinel/infer.py` | Provider-agnostic SLM calls; 3-tier degradation |
| `src/fraud_sentinel/guardrails.py` | 4 post-inference override rules |
| `src/fraud_sentinel/evaluate.py` | Adversarial suite + metrics |
| `scripts/run_pipeline.py` | End-to-end run |
| `scripts/validate_submission.py` | Independent 10-check schema audit |
| `scripts/compare_models.py` | Base vs fine-tuned, identical inputs |
| `solution.ipynb` | Submission notebook (22 cells) |

Nothing is dropped silently: every rejected row lands in `outputs/quarantine.csv` with the
rule that rejected it.

---

## Extending this

| To add | Change | Propagates to |
|---|---|---|
| A risk signal | one boolean column + one `WEIGHTS` entry | scoring, prompts, labels, training |
| A model backend | implement `LLMProvider` | the whole pipeline |
| A guardrail | append a `Rule` to `RULES` | every prediction |
| A new input schema | `TEXT_COLUMNS` / `_col()` in `clean.py` | ingestion |

Real labels, when they arrive, replace `rule_label` as the training target with no
structural change.

---

## Requirements

Python 3.12, `uv`, and **any OpenAI-compatible server** hosting a sub-3B model (see
[Serving the model](#serving-the-model)). Fine-tuning with `scripts/finetune.sh`
additionally needs Apple Silicon, since it uses MLX; the pipeline itself is
platform-agnostic. Without a reachable endpoint the pipeline degrades to the rule engine
rather than failing — `--no-slm` makes that explicit.
