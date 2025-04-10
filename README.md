# LLM-fine-tune
This project fine-tunes a decoder-only language model to generate structured JSON feedback from speech transcripts. Each output includes per-trait scores (1–10) and explanations for traits like formality, empathy, enthusiasm, and filler words.

Data is labeled using GPT-4o-mini. Evaluation ensures JSON correctness and trait-level consistency.

⸻

Next Steps
	•	Use GPT-4o-mini + chat history to generate diverse low-score examples
	•	Define score bins and identify rare cases per trait
	•	Upsample or augment rare-score samples
	•	Finalize trait-aware loss configuration (structure + weighted MSE)
	•	Train and evaluate with per-trait metrics on validation set

    Fine-tune
    •	If the base model already produces well-structured outputs → 
    1-stage fine-tuning with:
	•	Cross-entropy loss (token prediction)
	•	(JSON structure penalty)
	•	(Weighted score regression loss)
	•	If formatting is unreliable → 2-stage fine-tuning:
	1.	Structure alignment (JSON validity)
	2.	Score accuracy (weighted MSE)