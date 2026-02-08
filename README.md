# LLM-Fine-Tuning-Lab
---
### About 🧩: 
- In this repo, I play around with fine-tuning.  Each different folder is a different project.  

---
| Project | Task Type | Training Objective | Fine-Tuning Method | Key Techniques | Evaluation Focus |
|--------|----------|--------------------|--------------------|----------------|------------------|
| **01: Fraud Detection** | Binary classification (fraud vs non-fraud) | Supervised Fine-Tuning (SFT) | PEFT (QLoRA + LoRA adapters, 4-bit) | • Tabular → text prompt conversion<br>• Severe class imbalance handling via upsampling<br>• Decision threshold tuning (precision–recall)<br>• Prompt-based classification | Recall-heavy evaluation for minority class (fraud detection), confusion matrix analysis |
| **02: USMLE MCQ Reasoning** | Multi-class classification (4-choice MCQ) | Supervised Fine-Tuning (SFT) | PEFT (QLoRA + LoRA adapters, 4-bit) | • Logit-based answer scoring (final-token logits)<br>• Per-class upsampling for imbalance<br>• Instruction-tuned base model<br>• Cosine LR schedule + gradient checkpointing | Macro metrics (accuracy, precision, recall, F1) on held-out questions |
| **03: Emotion Classification** | Multi-class classification (6 emotions) | Supervised Fine-Tuning (SFT) | PEFT (QLoRA + LoRA adapters, 4-bit) | • Prompt-based constrained next-token classification<br>• Class-weighted CrossEntropyLoss (no resampling)<br>• Custom Trainer overriding loss<br>• Logit masking over label tokens | Macro F1 and recall across minority classes, confusion matrix balance |


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
- **Highlights:**
    - Scoring performed using final-token logits for A/B/C/D options
    - ⚖️ Handling Class Imbalance ⚖️ - Class frequencies were uneven, so the dataset was **upsampled with per-class balancing** to prevent the model from overpredicting the majority label.
    - QLoRA + 4-bit quantization
    - SFTTrainer (TRL)
    - LoRA config: r=16, α=32, dropout=0.05
    - Training took place on GPU using cosine LR schedule & gradient checkpointing
- **📊 Baseline vs Fine-Tuned Performance (first 200 eval samples):**
    - Model meaningfully improved after tuning
    - Even small fine-tuning (~3 epochs) improved classification consistency and accuracy
    - Confusion matrix became more evenly distributed across classes.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|----------------|--------------|----------|
| Baseline (untrained) | **0.615** | 0.660 | 0.619 | 0.597 |
| Fine-Tuned | **0.650** ↑ | 0.650 | 0.648 | 0.647 |

---

### ⭐ 03: Fine-Tuning an LLM for Emotion Classification (Class-Weighted QLoRA) 🎭

- **Project Overview:** I fine-tuned `Meta-Llama-3.1-8B-Instruct` to perform **multi-class emotion classification** on short text inputs using the `dair-ai/emotion` dataset.  The task involves predicting one of **six emotions**: *sadness, joy, love, anger, fear, surprise*.  Rather than treating this as a traditional sequence classification problem, I framed it as a **next-token prediction over constrained answer options**, leveraging the LLM’s generative nature while retaining a clean classification interface.
- **Dataset** 🔗 https://huggingface.co/datasets/dair-ai/emotion
- **Highlights:**
  - ⚖️ **Class imbalance**
    - Some emotions (e.g., *surprise*, *love*) appear far less frequently than others (*joy*, *sadness*).
    - Instead of resampling the dataset, I addressed imbalance by:
        - Computing **inverse-frequency class weights**
        - Applying them directly inside a **custom `CrossEntropyLoss`**
        - Giving rare emotions higher penalty when misclassified
    - This encourages the model to pay attention to minority classes without distorting the data distribution.
- **📊 Baseline vs Fine-Tuned Performance (first 500 eval samples):**
- **Results & Interpretation:**
  - The baseline (unfine-tuned) model performed **poorly**, achieving only **28.4% accuracy** and a **macro F1 of 0.182**.
  - It collapsed to predicting a small subset of majority emotions (*joy* and *anger*), completely failing to recognize several classes (*sadness, fear, surprise*), as seen in the confusion matrix.
  - After fine-tuning with **QLoRA + class-weighted loss**, performance improved dramatically:
      - **Accuracy increased to 92.8%**
      - **Macro F1 improved from 0.196 → 0.893**
      - Minority emotions such as *love* and *surprise* achieved strong recall, indicating successful mitigation of class imbalance.
  - This demonstrates that:
      - Prompt-based next-token classification alone is insufficient without task-specific supervision
      - **Loss weighting is an effective alternative to resampling** for imbalanced multi-class NLP tasks
      - Even limited fine-tuning can transform a general LLM into a reliable, balanced classifier

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|----------------|--------------|----------|
| Baseline (unfine-tuned) | **0.284** | 0.196 | 0.279 | 0.182 |
| Fine-Tuned (QLoRA + class-weighted loss) | **0.928**   | 0.883 | 0.906 | 0.893 |

