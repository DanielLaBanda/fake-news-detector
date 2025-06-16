# 📰 Fake-News Detector with Transformer Models

This repository hosts a complete **Streamlit** application that classifies English-language news articles as **real (0)** or **fake (1)** using three fine-tuned transformer models (DistilBERT, BERT-base, ALBERT-base).  
The app also offers dataset exploration, basic Optuna tuning results and error analysis.

---

## 🚀 Live Demo  
👉 **<[https://fake-news-detector-ew6eylsn9cgfwsivjin4ce.streamlit.app/](https://fake-news-detector-ew6eylsn9cgfwsivjin4ce.streamlit.app/)>** &nbsp; 

---

## 📁 Repository  
👉 **<[https://github.com/DanielLaBanda/fake-news-detector](https://github.com/DanielLaBanda/fake-news-detector)>**

---

## 🧠 Models

| Model        | Base checkpoint              | Params | Zip size |
|--------------|-----------------------------|--------|----------|
| DistilBERT   | `distilbert-base-uncased`   | 66 M   | 120 MB |
| BERT-base    | `bert-base-uncased`         | 110 M  | 420 MB |
| ALBERT-base  | `albert-base-v2`            | 12 M   | 55 MB |

Because of their size, the fine-tuned weights aren’t committed to GitHub.  
They’re stored on Google Drive and are downloaded / unzipped automatically the first time the app runs.

📦 **Download folder (models + train CSV)**  
👉 <[https://drive.google.com/drive/folders/1yxKA8RkMeU7PVfer-5eZ95Oqhm7qNKqn?usp=drive_link](https://drive.google.com/drive/folders/15H6J5pO-xqnMABSI2Mfb9P1kops6simH?hl=es-419)> &nbsp;

---

## 🖼️ App Pages

1. **Inference Interface**  
   * Paste headline + text, pick a model, get prediction & probabilities.

2. **Dataset Visualization**  
   * Class balance, word-count histogram, word cloud (from `train_dataset.csv`).

3. **Hyperparameter Tuning**  
   * Shows Optuna F1-curve (5 trials).  
   * No deep search was performed — DistilBERT reached 0.98 F1 with default LR 3e-5 / batch 8, so time was spent on analysis instead.

4. **Model Analysis & Justification**  
   * Classification report, confusion matrix, error inspection.  
   * DistilBERT chosen for deployment (best F1, lowest latency & memory).

---

## 🧪 Quick test inputs

Copy **Title** and **Content** into the app to check predictions.

| Case | Scenario | Expected |
|------|----------|----------|
| 1 | EU green-hydrogen deal (formal) | Real |
| 2 | “Apple-seed cure cancer” click-bait | Fake |
| 3 | NASA pizza order on Mars (satire) | Fake |

<details>
<summary>Show text snippets</summary>

**Case 1 – Real**  
*Title:* EU Signs \$50 Billion Green-Hydrogen Deal With Morocco  
*Content:* BRUSSELS — The European Commission confirmed on Monday that it has signed a 15-year, €46 billion agreement with Morocco to import green hydrogen produced from Saharan solar farms. Commissioner Kadri Simson said the first shipments will arrive at the Port of Rotterdam in 2027 after new pipelines are completed. Analysts at Wood Mackenzie estimate the deal could cut Europe’s industrial carbon emissions by 2 percent.

---

**Case 2 – Fake**  
*Title:* Scientists Reveal Apple-Seed Extract CURES Stage-4 Cancer in 10 Days!  
*Content:* NEW YORK — In a “ground-breaking” study published on an unknown blog, researchers say an extract from common apple seeds eliminates all tumors in terminal cancer patients within ten days. The team, whose identities remain confidential “for safety,” claims pharmaceutical companies tried to bribe them with $100 million to keep the discovery secret. No peer-reviewed data or clinical trials were provided.

---

**Case 3 – Satire**  
*Title:* NASA Admits Mars Rover Ordered 12 000 Pizzas on Agency Credit Card  
*Content:* HOUSTON — In a press conference that stunned reporters, NASA accountants announced the Perseverance rover somehow placed a late-night order for 12 000 pepperoni pizzas to Jezero Crater. “We’re still figuring out how the robot learned DoorDash,” said mission lead Dr. Elena Cruz. NASA’s legal team is reportedly negotiating a refund with an interplanetary delivery fee of $87 billion.
</details>

---

## 🛠️ Local installation

```bash
git clone https://github.com/YOUR-USERNAME/fake-news-detector.git
cd fake-news-detector

# (optional) create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py

```

## 🙌 Built With

- 🤗 Hugging Face Transformers  
- 🔥 PyTorch  
- 🧼 Streamlit  
- 📊 Scikit-learn  
- 📈 Seaborn, Plotly, Matplotlib  
- 💻 Google Colab Pro (for training)

---

## 👤 Author

Developed by **[Daniel González - A01285898]([https://github.com/A01286211](https://github.com/DanielLaBanda))**  
For educational and demonstration purposes.

---
