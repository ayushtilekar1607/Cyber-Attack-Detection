# main page gui

import tkinter  as tk 
from tkinter import * 


from PIL import Image # For face recognition we will the the LBPH Face Recognizer 
from PIL import Image , ImageTk  

root = tk.Tk()
#------------------------------------------------------

root.configure(background="seashell2")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("GUI_main")
#------------------Frame----------------------



#-------function------------------------

def reg():
    
##### tkinter window ######
    
    print("reg")
    from subprocess import call
    call(["python", "registration.py"])   



def login():
    
##### tkinter window ######
    
    print("log")
    from subprocess import call
    call(["python", "login.py"])   
    


def window():
    root.destroy()

#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('slide.jpg')
image2 =image2.resize((w,h))

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)


lbl = tk.Label(root, text="Detecting Network Attack", font=('times', 30,' bold '), height=1, width=50,bg="BLACK",fg="white")
lbl.place(x=0, y=0)

#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
button1 = tk.Button(root, text='Login Now',width=15,height=1,bd=5,bg='dark blue',font=('times', 15, ' bold '),fg='white',command=login).place(x=1100,y=50)
button1 = tk.Button(root, text='Register',width=15,height=1,bd=5,bg='green',font=('times', 15, ' bold '),fg='white',command=reg).place(x=1100,y=100)
exit = tk.Button(root, text="Exit", command=window, width=15, height=1, bd=5,font=('times', 15, ' bold '),bg="red",fg="white").place(x=1100, y=150)

root.mainloop()
