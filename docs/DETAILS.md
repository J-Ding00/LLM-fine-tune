# Fine-Tuning Details and Evaluation Metrics

## Overview

This project fine-tunes a 0.5B parameter language model to generate structured JSON feedback for speech transcripts, with per-trait scoring and explanations.  
Fine-tuning was conducted using QLoRA (4-bit quantization) on chat-formatted inputs and outputs.

- **Dataset**: OpenWebText subset (cleaned and structured)
- **Data Augmentation**: Minority trait classes were synthetically augmented using GPT-4o-mini
- **Labeling Source**: GPT-4o-mini
- **Fine-Tuning Method**: QLoRA (8-rank adapters, NF4 quantization, float16 training)

---

## Evaluation Process

- **Pretrained Model**: Evaluated both full and quantized outputs against ground truth labels.
- **Fine-Tuned Models**: Evaluated three fine-tuned checkpoints (epoch 1, 2, 3) against ground truth labels.
- **Metrics**:
  - Mean Squared Error (MSE) for each trait
  - Overall average MSE
  - JSON format validity rate
- **Test Sets**:
  - Common subset (samples valid across all models)
  - Individual valid subset (samples valid per model)

All MSE comparisons are made against ground truth labels.

---

## Results Summary

| Metric | Pretrained (Quantized) | Fine-Tuned (Best Epoch) |
|:---|:---|:---|
| Avg MSE | 8.56 | 1.99 |
| JSON Validity | ~58% | >99% |

---

## Detailed Metrics

### Common Subset Evaluation

MSE comparison on the same set of valid test samples:

| Trait | Pretrained | Fine-Tuned (Best) |
|:---|:---|:---|
| Formality | 4.94 | 2.07 |
| Persuasiveness | 6.94 | 1.70 |
| Enthusiasm | 8.51 | 1.58 |
| Empathy | 8.49 | 2.19 |
| Filler Words | 11.01 | 3.69 |
| Transition Logic | 10.97 | 1.04 |

---

### Individual Subset Evaluation

Each model evaluated on its own valid parsed samples:

| Model | Overall Avg MSE |
|:---|:---|
| Pretrained (Quantized) | 8.48 |
| Fine-Tuned Epoch 1 | 2.38 |
| Fine-Tuned Epoch 2 | 2.72 |
| Fine-Tuned Epoch 3 | 2.05 |

---

## Experiment Tracking

Training and evaluation logs were tracked using [Weights & Biases](https://wandb.ai/jcdingjobs-independent/LLM-fine-tune/runs/dg968a8f?nw=nwuserjcdingjobs).

---

## Notes

- Evaluation focused on MSE between model output scores and ground truth scores.
- Only samples with valid parsable JSON outputs were used for metric calculation.
- Full evaluation scripts and logs are available in the `eval/` directory.