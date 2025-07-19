# iris_classifier_gui_complete.py

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv("Iris.csv")  # Ensure Iris.csv is in the same directory
df.drop("Id", axis=1, inplace=True)  # Drop useless Id column

# Step 3: Data exploration
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Step 4: Data visualization
sns.pairplot(df, hue="Species")
plt.suptitle("Feature Relationships by Species", y=1.02)
plt.show()

# Step 5: Prepare data for model
X = df.drop("Species", axis=1)
y = df["Species"]

# Encode species labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix")
plt.show()

# Train/Test accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Cross-validation
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

# Save model accuracy for GUI
model_accuracy = accuracy * 100

# Step 8: Build GUI with Tkinter
def predict_species():
    try:
        sl = float(entry_sl.get())
        sw = float(entry_sw.get())
        pl = float(entry_pl.get())
        pw = float(entry_pw.get())

        input_data = np.array([[sl, sw, pl, pw]])
        prediction = model.predict(input_data)
        species = le.inverse_transform(prediction)[0]

        label_result.config(text=f"Predicted Species: {species}")
        label_accuracy.config(text=f"Model Accuracy: {model_accuracy:.2f}%")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

def reset_fields():
    entry_sl.delete(0, END)
    entry_sw.delete(0, END)
    entry_pl.delete(0, END)
    entry_pw.delete(0, END)
    label_result.config(text="Predicted Species:")
    label_accuracy.config(text="Model Accuracy:")

# GUI layout
root = Tk()
root.title("Iris Species Predictor")
root.geometry("400x420")
root.configure(bg="Moccasin")

Label(root, text=" Iris Species Predictor", font=("Arial", 16, "bold"), bg="Moccasin").pack(pady=10)

# Input fields
def create_input(label_text):
    frame = Frame(root, bg="Moccasin")
    frame.pack(pady=5)
    Label(frame, text=label_text, width=18, anchor='w', bg="Moccasin", font=("Arial", 12)).pack(side=LEFT)
    entry = Entry(frame, font=("Arial", 12), width=15)
    entry.pack(side=LEFT)
    return entry

entry_sl = create_input("Sepal Length (cm):")
entry_sw = create_input("Sepal Width (cm):")
entry_pl = create_input("Petal Length (cm):")
entry_pw = create_input("Petal Width (cm):")

# Buttons
frame_buttons = Frame(root, bg="Moccasin")
frame_buttons.pack(pady=15)
Button(frame_buttons, text="Predict Species", command=predict_species, font=("Arial", 12), bg="PeachPuff").pack(side=LEFT, padx=10)
Button(frame_buttons, text="Reset", command=reset_fields, font=("Arial", 12), bg="PeachPuff").pack(side=LEFT)

# Result
label_result = Label(root, text="Predicted Species:", font=("Arial", 12), bg="Moccasin")
label_result.pack(pady=10)

label_accuracy = Label(root, text="Model Accuracy:", font=("Arial", 12), bg="Moccasin")
label_accuracy.pack()

root.mainloop()
