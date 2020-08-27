import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import * 
from tkinter import filedialog 
from PIL import Image, ImageTk
from remove_bad_frames import *

def remove_frames(bool_list, fi_list, img2labelpath, csv_path):
    for bool_, fi in zip(bool_list, fi_list):
        if bool_ == "False":
            print(csv_path)
            print(img2labelpath)
            print(os.path.basename(fi))
            delete_labels(csv_path, img2labelpath, os.path.basename(fi))


def cont(bool_, window):
    if bool_:
        window.quit()
        for widget in window.winfo_children():
            widget.destroy()
            widget.pack_forget()


def make_buttons(values, window):
    bool_ = StringVar(window, "True")
    iter_ = 1

    for (text, value) in values.items(): 
        pady   = float(iter_) / 25
        rely   = 1 - pady
        Radiobutton(window,text=text,variable=bool_,  
                    value=value,indicator=False,background='grey').place(relx=0.5,rely=rely,anchor=S)
        iter_ += 1
    
    button = tk.Button(window,text="Continue",compound="bottom",background="grey",borderwidth=0,command=lambda : cont(True, window)).pack()
    
    mainloop()

    var = bool_.get()
    
    return var


def get_button_press(window):
    values       = {"Keep image for training library   " : "True", 
                    "Delete image from training library" : "False"} 

    button_press = make_buttons(values, window)

    return button_press


def preview_and_select(file, window):
    img          = Image.open(file)
    photo        = ImageTk.PhotoImage(img.convert('RGB'))
    label        = Label(image=photo)
    label.image  = photo
    label.pack()
    button_press = get_button_press(window)
    return button_press


def find_bad_frames(search_new_dir):

    while search_new_dir:

        window = tk.Tk()  
        window.title("Sort through labeled frames for all animals.")
        window.geometry("800x700")
        window.configure(background='grey')

        directory = filedialog.askdirectory()

        if directory == '':
            search_new_dir = False
            continue

        img2labelpath        = directory.split('_labeled')[0]
        csv                  = [f for f in os.listdir(img2labelpath) if os.path.isfile(os.path.join(img2labelpath, f)) and '.csv' in f][0]
        csv_path             = os.path.join(img2labelpath, csv)
        pngs                 = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and '.png' in f]
        full_path_pngs       = [os.path.join(directory, f) for f in pngs]
        bool_is_good_fi_list = [preview_and_select(f, window) for f in full_path_pngs]

        remove_frames(bool_is_good_fi_list, full_path_pngs, img2labelpath, csv_path)

        window.destroy()


if __name__ == "__main__":
    search_new_dir=True
    find_bad_frames(search_new_dir)
