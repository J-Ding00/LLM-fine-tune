# Fine-Tuning LLM for Structured Speech Feedback

This project fine-tunes a 0.5B parameter language model to generate **structured JSON feedback** with trait-based scoring and explanations for speech transcripts.

- **Data**: OpenWebText subset (realistic open-domain text samples)
- **Data Augmentation**: Minority trait samples synthetically generated using GPT-4o-mini
- **Labeling**: Ground truth labels generated using GPT-4o-mini
- **Model**: QLoRA (4-bit quantized) fine-tuning on chat-formatted prompts and outputs
- **Training Instance**: Single NVIDIA T4 GPU

## Results Overview
- MSE reduced from **8.56 → 1.99** after fine-tuning
- JSON structure validity improved from ~58% to **over 99%**
- Consistent improvement across multiple traits including formality, empathy, and logical transitions

**Detailed results and MSE comparison graphs are provided in [`docs/DETAILS.md`](docs/DETAILS.md).**