# 🧠 Emotion Text Classification with CNN

## 🎯 **Project Overview**
This project tackles the problem of **Emotion Detection** from text, utilizing **Convolutional Neural Networks (CNNs)**. The aim is to classify text data into seven emotions: **anger, fear, joy, love, neutral, sadness, and surprise**.

### 🔍 **Objective**
The objective of this project is to create an emotion classifier that can detect the emotional sentiment in user-generated text. The model is built using **CNNs** due to their ability to capture spatial hierarchies and patterns in the text.

---

## 📊 **Exploratory Data Analysis (EDA)**

Before building the model, we conducted a thorough **Exploratory Data Analysis (EDA)** to understand the distribution of emotions and the text structure.

- **Emotion Distribution**: The dataset consists of 7 classes with varying frequencies, which we handled using oversampling to balance the data.
- **Word Frequency**: We visualized the most common words in each class, providing insights into the unique words contributing to specific emotions.

### Key Insights:
- The dataset has a slight imbalance, with "neutral" and "love" being underrepresented.
- The most common words across emotions include "feel," "happy," and "sad."

---

## 🏗️ **Model Architecture**

We designed a dual-branch **Convolutional Neural Network (CNN)** to extract features from text. Here is a breakdown of the layers:

| Layer                      | Output Shape          | Parameters  |
|-----------------------------|-----------------------|-------------|
| **Input Layer**             | (None, 50)            | 0           |
| **Embedding Layer**         | (None, 50, 32)        | 256,000     |
| **Spatial Dropout 1D**      | (None, 50, 32)        | 0           |
| **Conv1D**                  | (None, 50, 64)        | 6,208       |
| **Batch Normalization**     | (None, 50, 64)        | 256         |
| **Global MaxPooling 1D**    | (None, 64)            | 0           |
| **Dense Layer**             | (None, 128)           | 16,512      |
| **Dropout**                 | (None, 128)           | 0           |
| **Output Layer**            | (None, 7)             | 903         |

### **Model Parameters**
- **Total Parameters**: 542,343
- **Trainable Parameters**: 542,087
- **Non-Trainable Parameters**: 256

The two branches capture different aspects of the text before being concatenated for the final classification.

---

## 🔧 **Model Training and Evaluation**

The model was trained using the following parameters:

- **Epochs**: 45
- **Batch Size**: 128
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

### **Model Performance**
The model achieved high accuracy and precision across most emotion classes:

| Emotion     | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **Anger**   | 0.81      | 0.76   | 0.79     | 1051    |
| **Fear**    | 0.86      | 0.76   | 0.81     | 1168    |
| **Joy**     | 0.95      | 0.92   | 0.93     | 1014    |
| **Love**    | 0.91      | 0.98   | 0.94     | 700     |
| **Neutral** | 0.83      | 0.97   | 0.89     | 701     |
| **Sadness** | 0.94      | 0.97   | 0.95     | 869     |
| **Surprise**| 0.70      | 0.74   | 0.72     | 717     |

**Overall Accuracy**: **86%**

---

## 📈 **Prediction Results**

We implemented the prediction mechanism using **Streamlit** to create a user-friendly interface. The app allows users to input their text and receive the predicted emotion along with a confidence score.

![Prediction Example]
<img width="680" alt="pop" src="https://github.com/user-attachments/assets/3a2614a8-44ea-4e6e-9d15-9259d0db5f47">


**Sample Output**:
- **Input**: "I am extremely happy today!"
- **Prediction**: **Joy**
- **Confidence**: **0.93**

### **Interactive Visualization**
We also plotted the probabilities for each emotion using a **bar chart**, enabling users to understand the model's predictions in a visually appealing format.

---

## 🔮 **Future Improvements**

- **Advanced Architectures**: Exploring LSTM or Transformer models for better sequence handling.
- **Data Augmentation**: Using NLP augmentation techniques to further improve model robustness.
- **Real-time Emotion Detection**: Deploying this model in real-time applications for chatbots and customer feedback systems.

---

## 💻 **Run the Project Locally**

To run the project on your local machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/YourUsername/emotion-classification-cnn.git
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

---

## 🔗 **References**

- **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Streamlit**: [https://streamlit.io/](https://streamlit.io/)

Feel free to reach out for collaborations or questions!

---
