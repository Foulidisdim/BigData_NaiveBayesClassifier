# 📦 Amazon Product Review Sentiment Analysis with PySpark and Spark NLP

> 💡 Developed as part of a Big Data coursework at the University of Macedonia, Thessaloniki, November 2024.

---

## 👥 Authors

- **Paraskevi Tsormpari**
- **Foulidis Dimitrios** 

---

## 🔍 Overview

This project performs **sentiment analysis** on a large set of Amazon product reviews using the **Apache Spark** framework with **PySpark** and **Spark NLP**. The goal is to classify reviews as **positive** (rating 4-5) or **negative** (rating 0-2) based on the text of the reviews. It uses a **Naive Bayes classifier** for sentiment prediction after applying various text preprocessing steps such as tokenization, stop word removal, lemmatization, and stemming.

This project was run on a **distributed Spark cluster** with multiple virtual machines to process large amounts of data efficiently.

---

## 🧰 Tech Stack

- **Apache Spark** (PySpark)
- **Spark NLP** (Natural Language Processing Library)
- **Naive Bayes Classifier**
- **Python 3**
- **HDFS** (Hadoop Distributed File System) and YARN (Yet Another Resource Negotiator)

---

## ⚙️ Prerequisites

- Apache Spark and Hadoop must be installed on the machine(s) for distributed execution.
- The Spark NLP library must be installed.
- Ensure **Python 3** and required libraries (PySpark, NumPy, Spark-NLP) are installed.

```bash
pip install pyspark
pip install spark-nlp
```

---

## 🧪 Data Flow

### 1. **Data Ingestion**

The dataset is a **JSONL file** containing Amazon product reviews with the following columns being of essence:
- `asin`: Unique review identifier.
- `text`: Review text.
- `rating`: Review rating (0-5).

### 2. **Data Cleaning & Preprocessing**
- **Trim** and **lowercase** the review text to ensure uniformity.
- **Filter out** reviews with a score of 3 (neutral).
- **Label reviews** as negative (0) or positive (1) based on the rating.
- **Remove duplicate reviews** using the `asin` and `text` columns.

### 3. **Text Processing (Using Spark NLP)**

- **Tokenization**: Split the text into individual words.
- **Stop Words Removal**: Eliminate common words that don't contribute to sentiment analysis.
- **Lemmatization**: Convert words to their base form (e.g., "running" -> "run").
- **Stemming**: Reduce words to their root form (e.g., "studies" -> "study").

### 4. **Feature Extraction**

- **Count Vectorization**: Convert the processed text into feature vectors (word counts).

### 5. **Model Training and Evaluation**

- Split the data into **training** and **testing** datasets.
- Use a **Naive Bayes classifier** to train the model based on the features extracted from the review text.
- Evaluate the model using **precision** and **recall** metrics.

---

## 🖥️ How to Run

1. **Set up your environment**:
    - Make sure that **Spark** and **Spark NLP** are installed and configured.
    - Ensure the necessary libraries are installed in your Python environment.
    - For disributed execution, Connect to each VM via SSH, update packages, install Python, NumPy, Spark NLP, and Java,         and set up passwordless SSH for communication between master and slave nodes. Install Hadoop, configure HDFS, and set 
    environment variables. Install and configure Spark, and adjust the spark-env.sh and workers files for distributed           execution. Finally, ensure sufficient disk storage and start Hadoop's DFS and YARN services, upload input data to HDFS,
    and launch the Spark cluster with spark-submit to run distributed tasks across the nodes.

2. **Run the script**:
   The Python script is executed from the command line with the following arguments:
   - `input_path`: Path to the input JSONL file containing the reviews.
   - `train_ratio`: Ratio of data to use for training (e.g., 0.8).
   - `test_ratio`: Ratio of data to use for testing (e.g., 0.2).

```bash
Example: spark-submit sentiment_analysis.py /path/to/reviews.jsonl 0.8 0.2
```

3. **Output**:
   The script will output the following metrics:
   - Precision
   - Recall

---

## 📈 Performance

The performance metrics (precision and recall) will be printed after the execution of each model training and evaluation run. These metrics help assess how well the Naive Bayes classifier performs on the given dataset.

---

## 📚 License

This project is developed for educational purposes and is not intended for commercial use.
