# Speech Feedback Fine-Tuning Pipeline

This project fine-tunes a decoder-only language model to generate structured JSON feedback (with scores and short explanations) on speech transcripts. Each transcript is assessed across multiple communication traits.

## Overview

The model is trained to evaluate transcripts and output a structured JSON object covering the following traits:

- Formality  
- Persuasiveness  
- Enthusiasm  
- Empathy  
- Filler Words  
- Transition Logic

### Fine-tuning Strategy

- **Stage 1**: Teach the model to consistently output the correct JSON format  
- **Stage 2**: Improve accuracy of the trait scores using score-based loss

## Workflow

1. Sampled text from OpenWebText as base input
2. Used gpt-4o-mini to generate synthetic low-quality examples for underrepresented traits
3. Labeled all samples with gpt-4o-mini for trait scores and feedback
4. Validated and cleaned outputs to ensure proper JSON structure
5. Split data into train/val/test with stratification
6. Fine-tuned a pretrained LLM using standard causal LM loss

## Usage

Run the following steps in order:

1. **Generate and label data**  
   `generate_data.py` and `gt_label_sample.py` / `pretrain_label_sample.py`

2. **Clean and validate labeled data**  
   `clean_data.py` and `evaluate_jsonl_format.py`

3. **Split and preprocess the dataset**  
   `split_data.py` and `preprocess.py`

4. **Fine-tune the model**  
   `fine_tune.py`

## Status
 
- Stage 1 fine-tuning in progress