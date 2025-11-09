# LLM-Fine-Tuning-Lab
---
## 01: Fine-Tuning an LLM for Fraud Detection 💸
- **Project Overview** - In this module, I fine-tuned an LLM to detect fradulent transactions using a highly imbalanced dataset.  This data set comes from a credit card fraud detection source, where fraudulent cases are extremely rare compared to legitimate transactions.
- 🔗 **Credit Card Fraud Detection Dataset** - https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection
- **Challenge:** Imbalanced Classes - One of the main challenges was the severe class imbalance. In the original dataset, only about 0.17% of the transactions were labeled as fraudulent. This caused the model to default to always predicting the majority class (legitimate transactions) in order to achieve a superficially high accuracy.  To overcome this, I needed to rebalance the dataset. By upsampling the positive (fraudulent) class to a more reasonable ratio—around 10% of the training data—I helped the model learn to identify fraud cases more effectively. This rebalancing step is crucial because it prevents the model from ignoring the minority class entirely.
- **Highlights:** 
    - **Converted Tabular Data to Text Prompts:** Each transaction was turned into a natural-language prompt so that the LLM could handle it as a text classification task.
    - **Rebalanced the Training Set:** We increased the proportion of fraudulent samples in the training data to around 10% to give the model a better chance to learn those patterns.
    - **Adjusted the Decision Threshold:** Instead of relying on a default 0.5 threshold, I tuned the threshold based on the precision-recall curve to better catch the minority class.
- **Results and Takeaways**
    - By rebalancing the dataset and adjusting the threshold, we improved the model’s ability to detect fraudulent transactions. This module highlights the importance of handling class imbalance in any fraud detection task and shows how to leverage LLMs for this kind of problem.
---
