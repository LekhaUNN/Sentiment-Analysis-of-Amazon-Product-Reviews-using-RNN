# 🛒 Sentiment Analysis of Amazon Product Reviews using RNN

A NLP project that classifies Amazon product reviews into sentiment ratings (1–5) using Recurrent Neural Networks. Compares SimpleRNN and LSTM architectures on 25,000 customer reviews.

---

## 📌 Overview

This project applies Natural Language Processing (NLP) and sequence modeling to predict the sentiment rating behind Amazon product reviews. Review text is cleaned, tokenized, and padded before being fed into RNN-based models for 5-class sentiment classification.

**Dataset Source:** [Amazon Product Review Dataset – Kaggle](https://www.kaggle.com/)


## 🎯 Sentiment Classes

| Label | Meaning       |
|-------|---------------|
| 1     | Very Negative |
| 2     | Negative      |
| 3     | Neutral       |
| 4     | Positive      |
| 5     | Very Positive |


## 📊 Dataset Summary

- **Total Samples:** 25,000
- **Columns:** Review (text), Sentiment (1–5)
- **Missing Values:** 1 (dropped)
- **Class Distribution:** ~5,000 samples per sentiment class (balanced)


## ⚙️ Workflow

```
Raw CSV Data
    ↓
EDA — null check, class distribution
    ↓
Text Cleaning
  • Remove HTML tags, special characters, numbers
  • Lowercase conversion
  • Tokenization (NLTK word_tokenize)
  • Stop word removal
    ↓
Text Encoding
  • Keras Tokenizer → integer sequences
  • Padding to max_words = 500
  • One-hot encoding of labels
    ↓
Train / Test Split  (80 / 20)
    ↓
Model 1 — SimpleRNN
Model 2 — LSTM
    ↓
Evaluation — accuracy, loss, confusion matrix, classification report
```

## 🏗️ Model Architectures

### Model 1 — SimpleRNN

```
Input          → shape (500,)
Embedding      → vocab_size × 500    [19,052,500 params]
SimpleRNN      → 128 units, relu, return_sequences=True
SimpleRNN      → 64 units, relu
Dense (output) → 5 units, softmax
```

### Model 2 — LSTM

```
Input          → shape (500,)
Embedding      → vocab_size × 128    [4,877,440 params]
LSTM           → 150 units
BatchNorm + Dropout(0.5)
Dense          → 50 units, relu
BatchNorm + Dropout(0.5)
Dense (output) → 5 units, softmax
```

Both compiled with: `loss = categorical_crossentropy`, `optimizer = Adam`


## 📈 Results

| Model     | Test Accuracy | Test Loss |
|-----------|--------------|-----------|
| SimpleRNN | 42.36%       | 1.3093    |
| LSTM      | 45.78%       | 1.3079    |

### LSTM Classification Report

| Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 1         | 0.59      | 0.51   | 0.54     |
| 2         | 0.39      | 0.42   | 0.41     |
| 3         | 0.37      | 0.40   | 0.38     |
| 4         | 0.40      | 0.41   | 0.40     |
| 5         | 0.57      | 0.55   | 0.56     |
| **Weighted Avg** | **0.47** | **0.46** | **0.46** |

> Note: 5-class sentiment rating is a harder task than binary classification. Mid-range ratings (2, 3, 4) are inherently ambiguous in text, which explains lower scores for those classes.


## 🔍 Sample Predictions

```python
predict_review_rating('Worst product')
# → Rating: 1

predict_review_rating('Awesome product, I will recommend this to other users.')
# → Rating: 5
```

## 🛠️ Requirements

```bash
pip install numpy pandas nltk matplotlib seaborn wordcloud textblob tensorflow keras
```

| Library    | Purpose                              |
|------------|--------------------------------------|
| numpy      | Numerical operations                 |
| pandas     | Data loading & manipulation          |
| nltk       | Tokenization, stopword removal       |
| matplotlib | Training curve plots                 |
| seaborn    | Confusion matrix heatmap             |
| wordcloud  | Word frequency visualization         |
| textblob   | Text processing utilities            |
| tensorflow | Neural network framework             |
| keras      | High-level model API (RNN, LSTM)     |


## 🚀 How to Run

1. **Clone this repository**
   ```
   Download ZIP or clone via GitHub
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (runs automatically in notebook)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. **Download the dataset** from Kaggle and place it in the project folder:
   ```
   Amazon_Product_Dataset.csv
   ```

5. **Run the notebook**
   ```
   Open sentiment_analysis_rnn.ipynb in Jupyter Notebook
   ```

## 📁 Project Structure

```
amazon-sentiment-rnn/
│
├── sentiment_analysis_rnn.ipynb   # Main Jupyter notebook
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔮 Future Improvements

- Increase epochs for better convergence (only 2–3 used here)
- Try pre-trained embeddings (GloVe, Word2Vec) instead of learned embeddings
- Use bidirectional LSTM for richer context capture
- Experiment with BERT for state-of-the-art NLP performance
- Add Flask API for live review rating prediction

