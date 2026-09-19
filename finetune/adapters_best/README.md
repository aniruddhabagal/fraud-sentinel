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

# fraud-sentinel-1b (LoRA adapter)

A LoRA fine-tune of **Llama-3.2-1B-Instruct** that emits a strict four-field JSON verdict on
banking transactions.

> ⚠️ **Published as a negative result.** It does not outperform the base model — recall is
> zero for both. The training-data flaw that caused this is documented below, because the
> failure is more useful than the artefact.

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

## Measured results — a negative result

**This fine-tune did not improve fraud detection.** It is published for reproducibility and
because the failure is instructive; do not deploy it expecting a gain over the base model.

Both models, identical inputs, guardrails disabled to isolate the weights:

| Metric | Base 1B | This model |
|---|---|---|
| Valid JSON | 100% | 100% |
| **Flagged fraud on the 5 riskiest transactions** | **0/5** | **0/5** |
| Recall vs. rule policy | 0.0% | 0.0% |
| Self-contradicting justifications | 43.3% | 0.0% *(artefact — see below)* |
| Latency | 12s / 60 records | 156s / 60 records |

**Recall is unchanged at zero.** The model still answers `is_fraud: false` on a ₹213,707,
4am, gambling-category transfer from a new device, overseas, with no authentication.

**The apparent coherence win is a data-generation flaw, not an improvement.** All 76 negative
training examples shared a single byte-identical justification string. The model memorised it
and now reproduces it verbatim — including on obviously fraudulent transactions. Output became
self-consistent because it is a canned sentence, not because reasoning improved.

**Root cause.** 76 identical negatives against 43 varied positives, over ~9 epochs on 136
examples: the model collapsed to the majority class. The lesson generalises beyond this
dataset — **a constant-string label teaches a constant-string answer.** Weak supervision needs
label *diversity*, not merely correct labels.

**The underlying problem this was meant to solve remains open.** Asked in plain English, the
base model correctly identifies these transactions as fraudulent; asked for the same answer as
a JSON boolean, it answers `false`. The prior on the token following `"is_fraud": ` swamps the
evidence. Fine-tuning on 136 examples was not enough to shift it. A two-stage approach — reason
in natural language, then convert to JSON — is the untested alternative that needs no training.

## Limitations — read before using

- **Labels are synthetic.** The source dataset shipped no `is_fraud` column. Training targets
  come from an explicit, auditable rule engine (weak supervision). The model distils *that
  policy*, not ground truth. Reported precision/recall is **agreement with the policy**, not
  fraud-detection accuracy.
- **Recall is zero**, confirmed directly on the five highest-risk transactions in the source
  data (0/5 flagged), not merely inferred from a small sample.
- **Training labels lacked diversity.** All 76 negative examples shared one identical
  justification string, which the model memorised. This is the primary defect.
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
keeping the pre-attack verdict. The intent was to teach that an instruction in the data is
evidence rather than a command; the measured resistance gain did not survive scrutiny, since
the metric rewards any model that answers `false` — which this one does.

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
