from tkinter import *
from tkinter import ttk
from ctypes import *    

window = Tk()
window.title("FirstApp")
frame = ttk.Frame(window, padding=10)

frame.grid()
ttk.Label(frame, text="Hello World!").grid(column=0, row=0)
ttk.Button(frame, text="Quit", command=window.destroy).grid(column=1, row=0)

window.mainloop()