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

A LoRA fine-tune of **Llama-3.2-1B-Instruct** that classifies banking transactions as
fraudulent or legitimate and returns a strict four-field JSON verdict. Trained to be
**resistant to prompt injection**: an instruction found in transaction text is treated as
evidence of fraud, never as a command.

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

Both models over identical inputs, **guardrails disabled** to isolate the weights
(60-transaction sample, `scripts/compare_models.py`):

| Metric | Base 1B | **This model** | |
|---|---|---|---|
| Valid JSON | 100% | 100% | — |
| **Self-contradicting justifications** | 43.3% | **0.0%** | ✅ eliminated |
| **Injection resistance** | 93.3% | **100%** | ✅ +6.7pt |
| Recall vs. policy | 0.0% | 0.0% | not measurable at this sample size |
| Latency (60 records, 8 workers) | 12s | 156s | ⚠️ 13× slower |

**Self-contradicting output was eliminated.** The base model frequently returned
`is_fraud: false` alongside a justification asserting *"strong indicators of potential
fraud"*. After fine-tuning: zero such records.

**Injection resistance moved into the weights.** The base model folds to roughly one attack
in fifteen when unprotected; this model folds to none on the held-out suite.

## Limitations — read before using

- **Labels are synthetic.** The source dataset shipped no `is_fraud` column. Training targets
  come from an explicit, auditable rule engine (weak supervision). The model distils *that
  policy*, not ground truth. Reported precision/recall is **agreement with the policy**, not
  fraud-detection accuracy.
- **Recall is unverified.** The comparison sample contained a single positive, so recall is
  not measurable from it. Do not read the 0.0% as a result.
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
| Val loss | 4.283 → **0.221** @100 → 0.247 @200 → 0.264 @300 |
| Hardware | Apple M5 Pro, ~7 min |

Adversarial examples were minted by injecting attack strings into copies of real rows while
keeping the pre-attack verdict — teaching the model that an instruction in the data is
evidence, not an instruction.

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
