# all attack detection gui

from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from joblib import dump
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
root = tk.Tk()
root.title("All_Attack Detection")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.configure(background="black")

image = Image.open('s1.jpeg')

image = image.resize((1350,750))

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=70) #, relwidth=1, relheight=1)


lbl = tk.Label(root, text="All_attcak Detection", font=('times', 35,' bold '), height=1, width=62,bg="brown",fg="white")
lbl.place(x=0, y=0)


def Model_Training1():
    data = pd.read_csv("test.csv")
    data.head()

    data = data.dropna()

    """Feature Selection => Manual"""
    x = data.drop(['Average_Packet_Size', 'Duration','Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)

    # Replace SVM with Random Forest Classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=123)
    random_forest_classifier.fit(x_train, y_train)

    y_pred = random_forest_classifier.predict(x_test)
    print(y_pred)

    print("=" * 40)
    print("==========")
    print("Classification Report : ", classification_report(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))

    label4 = tk.Label(root, text=str(repo), width=45, height=15, bg='seashell2', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=250, y=100)

    label5 = tk.Label(root,
                      text="Accuracy : " + str(ACC) + "%\nModel saved as attack_RandomForest.joblib",
                      width=45, height=2, bg='black', fg='white', font=("Tempus Sanc ITC", 14))
    label5.place(x=250, y=420)

    dump(random_forest_classifier, "attack_RandomForest.joblib")
    print("Model saved as attack_RandomForest.joblib")
    

def call_file():
    from subprocess import call
    call(['python','Check1.py'])
    


def window():
    root.destroy()

button3 = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model_RF", command=Model_Training1, width=15, height=2)
button3.place(x=10, y=200)


button4 = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                    text="Check Performance", command=call_file, width=15, height=2)
button4.place(x=10, y=300)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=10, y=400)

root.mainloop()
