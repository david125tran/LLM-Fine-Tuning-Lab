# LLM-Fine-Tuning-Lab
---
### About 🧩: 
- In this repo, I play around with fine-tuning and apply concepts from the **[Hugging Face Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)** as well as two other tutorials that I learned fine-tuning from.  Each different folder is a different project.  

---
## ⭐ 01: Fine-Tuning an LLM for Fraud Detection 💸
- **Project Overview:** Fine-tuned a large language model to detect fraudulent transactions in a **highly imbalanced dataset** (fraud = ~0.17% of samples).
- **Dataset:** 🔗 https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection
- **Challenge:** Imbalanced Classes - One of the main challenges was the severe class imbalance. In the original dataset, only about 0.17% of the transactions were labeled as fraudulent. This caused the model to default to always predicting the majority class (legitimate transactions) in order to achieve a superficially high accuracy.  To overcome this, I needed to rebalance the dataset. By upsampling the positive (fraudulent) class to a more reasonable ratio—around 10% of the training data—I helped the model learn to identify fraud cases more effectively. This rebalancing step is crucial because it prevents the model from ignoring the minority class entirely.
- **Highlights:**
    - **Converted Tabular Data to Text Prompts:** Each transaction was turned into a natural-language prompt so that the LLM could handle it as a text classification task.
    - **Rebalanced the Training Set**
    - **Adjusted the Decision Threshold:** Instead of relying on a default 0.5 threshold, I tuned the threshold based on the precision-recall curve to better catch the minority class.
- **Results:**
    - The baseline LLM **failed at fraud detection**.  Fine-tuning transformed the model into an effective fraud detector.
   - Confusion Matrix Summary
        - **12 / 13** fraud cases correctly detected  
        - **Only 1** fraud case missed  
        - **32 false positives** out of ~5,000 transactions  
    - Final Model Performance (5,000 test samples)

| Metric | Result | Notes |
|---|---|---|
Accuracy | **0.993** | High, but not primary metric |
Precision | **0.273** | ~1 in 4 fraud predictions are correct (expected at this stage) |
Recall | **0.923** ✅ | Caught **92.3% of fraud cases** |
F1 Score | **0.421** | Balanced precision/recall for imbalanced data |

---

### **⭐ 02: Fine-Tuning an LLM for Medical USMLE MCQ Reasoning 🩺**

- **Project Overview:** Fine-tuned `Meta-Llama-3.1-8B-Instruct` on the **MedQA USMLE 4-choice multiple-choice dataset** to evaluate how well a general LLM can be adapted into a **medical reasoning classifier**.
- **Dataset:** 🔗 https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
- **Why Llama-3.1-8B-Instruct? 🧩** I chose the Llama-3.1-8B-Instruct model because it strikes a strong balance between capability and efficiency-large enough to capture nuanced medical reasoning, but still lightweight enough for practical fine-tuning with QLoRA on a single GPU. Its instruction-tuned architecture also makes it well-suited for structured tasks like multiple-choice reasoning and logits-based classification.
- **Highlights:**
    - Scoring performed by comparing **logits of final token** against option tokens
    - ⚖️ Handling Class Imbalance ⚖️ - Class frequencies were uneven, so the dataset was **upsampled with per-class balancing** to prevent the model from overpredicting the majority label.
    - QLoRA + 4-bit quantization
    - SFTTrainer (TRL)
    - LoRA config: r=16, α=32, dropout=0.05
    - Training took place on GPU using cosine LR schedule & gradient checkpointing
- **📊 Baseline vs Fine-Tuned Performance (first 200 eval samples):**
    - Model meaningfully improved after tuning
    - Even small fine-tuning (~3 epochs) improved classification consistency and accuracy

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|----------------|--------------|----------|
| Baseline (untrained) | **0.630** | 0.635 | 0.628 | 0.629 |
| Fine-Tuned | **0.680** ↑ | 0.676 | 0.675 | 0.672 |

---

