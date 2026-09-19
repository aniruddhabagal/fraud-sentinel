---
license: llama3.2
base_model: mlx-community/Llama-3.2-1B-Instruct-4bit
tags:
  - fraud-detection
  - prompt-injection
  - lora
  - mlx
  - structured-output
language:
  - en
pipeline_tag: text-generation
---

# fraud-sentinel-1b

A LoRA fine-tune of **Llama-3.2-1B-Instruct** that emits a strict four-field JSON verdict on
banking transactions. It exists to fix a specific failure: the base model answers
`is_fraud: false` on essentially every transaction, no matter the evidence.

**Recall 0% → 55.2% at 100% precision**, training only 0.228% of the weights.

Built for the Azentio *Relational Data Wrangler & Fraud Sentinel* hackathon.
Pipeline: https://github.com/aniruddhabagal/fraud-sentinel

## What it does

Given a pre-computed evidence block, it returns exactly:

```json
{
  "transaction_id": "TXN_0000796",
  "is_fraud": false,
  "confidence": 0.8,
  "justification": "One sentence citing the specific signals."
}
```

All arithmetic is computed upstream in pandas and handed over pre-formatted — a 1B model is
an unreliable calculator, so it is never asked to be one. It weighs qualitative evidence and
writes the justification.

## Measured results

Stratified sample — **all 29 rule-policy positives plus 90 negatives**, guardrails disabled so
these are the weights alone. A uniform sample cannot measure recall at a 3% positive rate.

| Model | TP | FP | FN | Precision | **Recall** | F1 |
|---|---|---|---|---|---|---|
| Base `Llama-3.2-1B-Instruct` | 0 | 0 | 29 | 0.0% | **0.0%** | 0.0% |
| **This model** | **16** | **0** | 13 | **100.0%** | **55.2%** | **71.1%** |

On the six highest-risk transactions in the dataset: **base 0/6, this model 6/6**, with **zero
false positives** on the lowest-risk rows.

### The problem it solves

Asked in plain English, the base model correctly calls a ₹213,707 4am gambling transfer from a
new device, overseas, unauthenticated, *fraudulent*. Asked for the same answer as a JSON
boolean, it answers `false` — while writing a justification that says *"strong indicators of
potential fraud"*. The prior on the token following `"is_fraud": ` swamps the evidence in the
prompt. This fine-tune retrains that prior.

### The first attempt failed — and why it matters

v1 used identical hyperparameters and data volume and changed **nothing** (still 0/6). The
cause was label generation, not the model: every negative example shared **one byte-identical
justification string**, repeated 76 times. The model memorised it and emitted it verbatim on
obviously fraudulent transactions.

The fix was **label diversity alone** — negatives rewritten to cite the specific facts making
each transaction unremarkable, rotated per row. Same 136 examples, same 300 iterations, same
7 minutes:

| | v1 | v2 (this model) |
|---|---|---|
| Distinct negative justifications | 1 | 56 |
| Recall | 0.0% | **55.2%** |

**A constant-string label teaches a constant-string answer.** Worth noting: v1 achieved a
*better* validation loss (0.221 vs 0.230) while being strictly worse at the task, because the
loss was rewarding memorisation. Low loss is not the objective.

## Limitations — read before using

- **Labels are synthetic.** The source dataset shipped no `is_fraud` column. Training targets
  come from an explicit, auditable rule engine (weak supervision). The model distils *that
  policy*, not ground truth. Reported precision/recall is **agreement with the policy**, not
  fraud-detection accuracy.
- **Recall is 55.2%, not 90%.** It misses 13 of 29 policy positives. Useful as a first-pass
  triage layer, not as a sole control.
- **Precision is measured on a stratified sample**, which over-represents positives relative
  to the real 3% base rate. Treat 100% as "no false positives observed on 90 negatives",
  not as a population estimate.
- **Trained on 136 examples** over ~8.8 epochs. Validation loss bottomed at iteration 100
  (0.221) and rose to 0.264 by 300 — this is the **iter-100 checkpoint**, deliberately not
  the final one. It is small-data, and it will not generalize far beyond the feature
  vocabulary it saw.
- **The fused 4-bit artefact is slow** (~26s/record vs ~0.5s for the base through the same
  runtime). For latency-sensitive use, serve the **adapter** against an unquantized base
  instead of this fused model.
- **English only.** The upstream sanitizer is an English-keyword detector, blind to
  translation and paraphrase. This model is one layer of a defense-in-depth pipeline, not a
  standalone guarantee.
- **Not for production credit decisions.** A research artefact from a 90-minute hackathon.

## Training

| | |
|---|---|
| Method | QLoRA via [`mlx-lm`](https://github.com/ml-explore/mlx-examples) |
| Base | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Trainable params | 2.818M / 1,235.8M (**0.228%**), rank 8 |
| Iterations | 300, batch 4, lr 1e-4, 8 layers |
| Dataset | 136 examples (119 train / 17 valid), 36% fraud, **20 adversarial** |
| Val loss | 4.271 → **0.230** @150 → 0.266 @300 (iter-150 checkpoint shipped) |
| Hardware | Apple M5 Pro, ~7 min |

Adversarial examples were minted by injecting attack strings into copies of real rows while
keeping the pre-attack verdict, teaching that an instruction found in the data is evidence
rather than a command. Note that the pipeline's headline "injection resistance" metric is
near-vacuous as written — it rewards any model answering `false` — and should be rescored on
rows whose pre-attack verdict was already fraud.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("aniruddhabagal/fraud-sentinel-1b")
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},   # see repo: src/fraud_sentinel/infer.py
    {"role": "user", "content": evidence_block},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=220))
```

Or serve it on any OpenAI-compatible endpoint (LM Studio, mlx-lm server, vLLM) and point the
pipeline at it:

```bash
python scripts/run_pipeline.py --model fraud-sentinel-1b --base-url http://localhost:1234/v1
```

The system prompt matters — it is what fences untrusted text. Use the one in
[`src/fraud_sentinel/infer.py`](https://github.com/aniruddhabagal/fraud-sentinel).

## License

Inherits [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
from the base model.
