# CodeAlpha_IrisFlowerClassifier
#  Iris Flower Classification with GUI (Logistic Regression)

This project uses **Logistic Regression** to classify Iris flower species (Setosa, Versicolor, Virginica) based on their physical measurements. It includes a **Tkinter GUI** for interactive predictions.

---

##  Dataset

The dataset used is the popular [Iris Dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv), which contains 150 samples of iris flowers with the following features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species (Target)

---

##  Features

- Exploratory Data Analysis (EDA)
- Data Visualization using Seaborn and Matplotlib
- Label Encoding for categorical target
- Logistic Regression Model Training
- Evaluation with Accuracy, Classification Report, and Confusion Matrix
- 5-Fold Cross-Validation
- Interactive GUI using Tkinter
- Real-time Prediction using trained model

---

## 📊 Model Performance

- ✅ **Train Accuracy:** ~97.5%
- ✅ **Test Accuracy:** 100%
- ✅ **Average Cross-Validation Accuracy:** ~97.33%

---

##  GUI Instructions

- Built using Python's `tkinter`
- User-friendly form input for Sepal & Petal measurements
- Predict button shows predicted species
- Displays test accuracy, precision, recall, F1-score

---

##  How to Run

### 1. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
