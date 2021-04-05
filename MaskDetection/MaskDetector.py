import tkinter
from Detector import *
from tkinter import filedialog
from tkinter.ttk import *
from tkinter import messagebox
from tkinter import *
import pkg_resources.py2_warn

def getFPS():
    if (detector.avarageFPS != None):
        averagefps = "Average FPS : {:.2f}".format((detector.avarageFPS))
        messagebox.showinfo("FPS",averagefps)
    else:
        messagebox.showwarning("No Video","Please play a video!!!")

def getTotalDetected():
    statu = "Nomask : " + str(detector.mask_status[1]) + " Mask : " + str(detector.mask_status[0])
    messagebox.showinfo("Statu",statu)

def browseVideoFile():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("Video Files","*.mp4*"),("all files","*.*")))
    file_Explorer.configure(text = "File Opened : " + filename)
    detector.detectInVideo(videoName= filename)

def browseImageFile():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("Image Files","*.jpg*"),("all files","*.*")))   
    file_Explorer.configure(text = "File Opened : " + filename)
    detector.detectInImage(imgName= filename)  

def LiveDetection():
    detector.detectInVideo(0)
    
root = tkinter.Tk()
root.geometry("800x600")
root.resizable(0,0)
root.title("Covid-19 Face Mask Detector")

st = Style()
st.configure('W.TButton', background='#345', foreground='black', font=('Arial', 14 ))

backgraundImg = tkinter.PhotoImage(file = "WindowBack.png")
IconImg = tkinter.PhotoImage(file = "FaceMaskIcon.png")
detector = faceDetector(use_cuda = True)
root.iconphoto(False,IconImg)

fps = tkinter.Button(root,text = "Get last average fps",relief=RIDGE,bd = "5",command = getFPS)
fps.pack(side = "top")
file_Explorer = Label(root, text = "No File",width = "50", height = 4,fg = "black")
file_Explorer.pack(side = "bottom")
img_Label = Label(root,image = backgraundImg)
img_Label.pack(side='top', fill='both', expand='yes')

Total = tkinter.Button(root,text = "GetTotalStatus",bd = '5',relief = RIDGE,command = getTotalDetected)
LiveButton = tkinter.Button(root,text="LiveDetection",bd = '5',relief=RIDGE,command = LiveDetection)
FromImageButton = tkinter.Button(root,text="DetectFromImage",bd = '5',relief=RIDGE,command = browseImageFile)
FromVideoButton = tkinter.Button(root,text="DetectFromVideo",bd = '5',relief=RIDGE,command = browseVideoFile)

Total.place(x = 342, y = 450)
LiveButton.place(x = 342, y = 350)
FromImageButton.place(x = 342, y = 250)
FromVideoButton.place(x = 342, y = 150)

root.mainloop()

