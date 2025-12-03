SQuAD Question Answering with DistilBERT
A PyTorch-based machine learning project that fine-tunes a pre-trained DistilBERT model on the Stanford Question Answering Dataset (SQuAD) to extract precise answers from given contexts.

📋 Overview
This project implements an extractive question answering system that takes a question and a context passage as input and returns the most relevant answer span from the passage. The model is trained on the SQuAD dataset, which contains 100,000+ question-answer pairs, and uses the powerful transformer architecture provided by Hugging Face's transformers library.

Key Features
Pre-trained Model: Leverages DistilBERT, a lightweight version of BERT with 40% fewer parameters

SQuAD Dataset: Trained on 10% of the SQuAD v1.1 dataset for efficient learning

Extractive QA: Identifies answer spans directly from the provided context

GPU/CPU Support: Automatically detects and uses GPU if available, otherwise falls back to CPU

Robust Preprocessing: Handles tokenization, truncation, and proper mapping of character-level answers to token-level positions

Batch Processing: Efficiently processes multiple examples using data collation and padding

🚀 Quick Start
Prerequisites
Ensure you have Python 3.7+ installed. Install required dependencies:

bash
pip install torch torchvision torchaudio
pip install transformers datasets evaluate
pip install scikit-learn numpy pandas
Installation
Clone or download this repository

Install dependencies (see Prerequisites)

Run the notebook or script

Usage
Training
The notebook provides a complete pipeline for training:

python
# Load dataset
dataset = load_dataset("squad", split="train[:10%]")

# Preprocess and split data
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
train_dataset = processed_dataset.select(range(train_size))
val_dataset = processed_dataset.select(rang