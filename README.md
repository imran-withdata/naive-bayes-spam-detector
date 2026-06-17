# SMS Spam Detection using Bag of Words (BoW)

This repository contains a step-by-step tutorial on building an NLP pipeline to classify SMS messages as either **Ham** (legitimate) or **Spam**.

## 📌 Project Overview

The project preprocesses raw text, converts messages into a structured numerical feature matrix using `CountVectorizer`, and trains a `MultinomialNB` classifier.

## 📊 Dataset

* **Total Records:** 5,572 records (4,825 Ham / 747 Spam)
* **Columns:** `Category` (Target label) and `Message` (Raw text)

## 🛠️ Pipeline Architecture

1. **Data Exploration:** Analyze target class frequencies.
2. **Train-Test Split:** Split into 80% training and 20% evaluation sets.
3. **Vectorization:** Apply Bag of Words via `CountVectorizer`.
4. **Model Training:** Train a Multinomial Naive Bayes classifier.
5. **Evaluation:** Generate precision, recall, and F1-scores.

## 🚀 Performance

* **Overall Accuracy:** 99%
* **Spam Precision:** 98%

## 💻 Setup & Usage

1. **Install requirements:** `pip install pandas numpy scikit-learn`
2. **Run:** Open `sms_spam_detector_bow.ipynb` in Jupyter, place your `spam.csv` dataset in the root directory, and execute the cells sequentially.
