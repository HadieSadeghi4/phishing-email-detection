# Phishing Email Detection using Logistic Regression & BERT

A dual-model NLP system for detecting phishing emails. Implements both a traditional **Logistic Regression** model with TF-IDF features and a deep learning model based on **DistilBERT**. Trained on the Enron Email Dataset, this project includes full preprocessing, training, evaluation, and an interactive demo for real-time phishing detection.

## 🔍 Key Features
- **Dual Model Approach**: Compare traditional ML vs. modern Transformer performance
- **Preprocessing Pipeline**: Full text cleaning (lowercasing, punctuation removal, etc.)
- **BERT-Based Training**: Fine-tuned DistilBERT with Hugging Face Transformers
- **Interactive Demo**: Test both models live with your own email text
- **Detailed Evaluation**: Includes classification reports, ROC curves, and loss graphs

## 🛠️ Built With
- `Python` • `pandas` • `scikit-learn`
- `Transformers` • `Hugging Face` • `PyTorch`
- `Matplotlib` • `Seaborn`

## 📁 Project Structure
project/
├── CSE-ARS.ipynb # Main Jupyter notebook
├── emails.csv # Dataset (Enron Emails)
└── README.md # Project description

## 🚀 Usage

To use this project, open and run the `CSE-ARS.ipynb` Jupyter notebook. The notebook includes the following main sections:

1.  **Data Preprocessing**
    - Load and clean the Enron email dataset
    - Remove headers, numbers, and punctuation
    - Convert text to lowercase
    - Label emails based on suspicious keywords

2.  **Model Training**
    - **Logistic Regression**: Train with TF-IDF features
    - **DistilBERT**: Fine-tune the transformer model for 10 epochs

3.  **Performance Evaluation**
    - Compare both models using classification reports
    - Generate ROC curves and loss graphs
    - Analyze precision, recall, and F1-score metrics

4.  **Interactive Demo**
    - Test both models with custom email text
    - Get real-time predictions for phishing detection

Simply execute all cells in the notebook sequentially to reproduce the entire workflow.

## 📊 Results

The project provides comprehensive evaluation metrics for both models:

### Logistic Regression
- Classification report with precision, recall, and F1-score
- ROC curve showing true positive vs. false positive rates
- Fast training and inference time

### DistilBERT
- Detailed classification metrics
- Training and validation loss graphs over 10 epochs
- Higher accuracy for detecting sophisticated phishing attacks
- ROC curve comparison with Logistic Regression

### Performance Comparison
- **Logistic Regression**: Faster training, suitable for baseline implementation
- **DistilBERT**: Higher accuracy, better at detecting complex phishing patterns

The visualizations include:
- Loss curves for BERT training
- ROC curves for both models
- Confidence scores for predictions

### Prerequisites
Make sure you have Python 3.7+ installed.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/HadieSadeghi4/phishing-email-detection.git
cd phishing-email-detection


