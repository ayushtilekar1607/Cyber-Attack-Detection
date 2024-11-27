# malware attack gui

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

root = tk.Tk()
root.title("malware")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))


image2 = Image.open('s4.jpg')

image2 = image2.resize((w, h))

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0) 


lbl = tk.Label(root, text="Malware Attack", font=('times', 35,' bold '), height=2, width=62,bg="violet Red",fg="Black")
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
    from joblib import dump
    dump (svcclassifier,"Malware_Model.joblib")
    print("Model saved as Malware_Model.joblib")



def ml():
    from subprocess import call
    call(['python','Check_Malware.py'])




def window():
    root.destroy()

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Training", command=Model_Training, width=15, height=2)
button3.place(x=5, y=200)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Malware Detection", command=ml, width=15, height=2)
button4.place(x=5, y=280)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=380)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''