# LLM-Fine-Tuning-Lab
---
## About: 
- In this repo, I play around with fine-tuning and apply concepts from the **[Hugging Face Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)**.  Each different folder is a different project.  
---
## 01: Fine-Tuning an LLM for Fraud Detection 💸
- **Project Overview** - Fine-tune a large language model to detect fraudulent transactions in a **highly imbalanced dataset** (fraud = ~0.17% of samples).
- 🔗 **Credit Card Fraud Detection Dataset** - https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection
- **Challenge:** Imbalanced Classes - One of the main challenges was the severe class imbalance. In the original dataset, only about 0.17% of the transactions were labeled as fraudulent. This caused the model to default to always predicting the majority class (legitimate transactions) in order to achieve a superficially high accuracy.  To overcome this, I needed to rebalance the dataset. By upsampling the positive (fraudulent) class to a more reasonable ratio—around 10% of the training data—I helped the model learn to identify fraud cases more effectively. This rebalancing step is crucial because it prevents the model from ignoring the minority class entirely.
- **Highlights:** 
    - **Converted Tabular Data to Text Prompts:** Each transaction was turned into a natural-language prompt so that the LLM could handle it as a text classification task.
    - **Rebalanced the Training Set**
    - **Adjusted the Decision Threshold:** Instead of relying on a default 0.5 threshold, I tuned the threshold based on the precision-recall curve to better catch the minority class.
- **Results**
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
