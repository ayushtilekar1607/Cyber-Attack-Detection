# gui to display attack

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
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

root = tk.Tk()
root.title("GUI")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('bg1.jpg')

image2 = image2.resize((w, h))

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Detecting Network Attack", font=('times', 25,' bold '), height=1, width=80,bg="black",fg="red")
lbl.place(x=0, y=0)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Model_Training():
    data = pd.read_csv("testing.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['ProductName'] = le.fit_transform(data['ProductName'])

    data['IsBeta'] = le.fit_transform(data['IsBeta'])
    data['AVProductStatesIdentifier'] = le.fit_transform(data['AVProductStatesIdentifier'])
    data['AVProductsInstalled'] = le.fit_transform(data['AVProductsInstalled'])
    data['AVProductsEnabled'] = le.fit_transform(data['AVProductsEnabled'])
    data['HasTpm'] = le.fit_transform(data['HasTpm'])
    data['CountryIdentifier'] = le.fit_transform(data['CountryIdentifier'])
    data['GeoNameIdentifier'] = le.fit_transform(data['GeoNameIdentifier'])
    data['Platform'] = le.fit_transform(data['Platform'])
    data['Processor'] = le.fit_transform(data['Processor'])
    data['OsPlatformSubRelease'] = le.fit_transform(data['OsPlatformSubRelease'])
       

    """Feature Selection => Manual"""
    x = data.drop(['HasDetections'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['HasDetections']
    print(type(y))
    x.shape
    

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1234)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as Malware_Model.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    dump (svcclassifier,"Malware_Model.joblib")
    print("Model saved as Malware_Model.joblib")



def call_file():
    from subprocess import call
    call(['python','GUI_Master_ml.py'])
    
def All():
    from subprocess import call
    call(['python','GUI_MASTER1.py'])
    
def phishing():
    from subprocess import call
    call(['python','phishing_attack.py'])


def window():
    root.destroy()


button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Malware_Attack", command=call_file, width=15, height=1)
button4.place(x=5, y=200)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="All_Attack", command=All, width=15, height=1)
button4.place(x=5, y=300)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Phishing_Attack", command=phishing, width=15, height=1)
button4.place(x=5, y=400)


button5 = tk.Button(root, text="Exit", command=window, width=15, height=1, font=('times', 14, ' bold '),bg="red",fg="white")
button5.place(x=5, y=500)

root.mainloop()
