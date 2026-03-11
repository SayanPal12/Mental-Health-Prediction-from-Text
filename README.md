# Mental Health Detection using LSTM

This project builds a **Deep Learning model using Long Short-Term Memory (LSTM)** networks to detect potential mental health indicators from textual data. The goal is to analyze written text and classify whether it reflects signs of mental distress or normal mental states.

With the increasing use of online platforms, people often express emotions, struggles, and psychological states through text. Machine learning models can help analyze such patterns and assist in **early detection of mental health signals**.

This repository demonstrates how **Natural Language Processing (NLP)** and **Recurrent Neural Networks (RNNs)** can be used to analyze textual data for mental health detection.

---

# Dataset

The dataset used in this project is publicly available on Kaggle:

Mental Status Dataset  
https://www.kaggle.com/datasets/footsurebead/mental-status

The dataset contains textual entries labeled with different mental health categories. These labels allow the model to learn patterns associated with various psychological states.

⚠️ This dataset is used strictly for **research and educational purposes**.

---

# Project Pipeline

The project follows the standard NLP deep learning workflow:

1. **Data Preprocessing**
   - Removing noise and unwanted characters
   - Lowercasing text
   - Tokenization
   - Stopword removal
   - Padding sequences

2. **Text Representation**
   - Word tokenization
   - Integer encoding
   - Sequence padding

3. **Model Architecture**
   - Embedding Layer
   - LSTM Layer(s)
   - Fully Connected Dense Layers
   - Softmax / Sigmoid Output Layer for classification

4. **Training**
   - Train-test split
   - Cross entropy loss
   - Adam optimizer

5. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

---

# Model Architecture

```
Input Text
   ↓
Text Preprocessing
   ↓
Tokenization & Padding
   ↓
Embedding Layer
   ↓
LSTM Layer
   ↓
Dense Layer
   ↓
Output Classification
```

The **LSTM network** helps capture sequential patterns and contextual dependencies in text, which are crucial for detecting emotional or psychological signals.

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)

---

# Disclaimer

This project is **not a medical diagnostic tool**. It is intended for research and educational purposes only. Mental health diagnosis should always be performed by qualified healthcare professionals.

---

# Future Improvements

- Use **Transformer-based models (BERT, RoBERTa)**
- Multi-label mental health classification
- Explainable AI for interpretability
- Deploy as a web application for real-time predictions

---

# Author

Developed as a Deep Learning project exploring **AI applications in mental health analysis**.
