import tkinter as tk

from  tkinter import StringVar, Label
from tkinter.messagebox import showerror
import tkinter
from PIL import ImageTk, Image
import pymysql
import cv2
import numpy as np
from scipy.misc.pilutil import imresize
#import cv2 #version 3.2.0
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import sys
# Backend code

global p
#connection = pymysql.connect(host="localhost", user="root", passwd="", database="Sahil")
#cursor = connection.cursor()

#This creates the main root of an application
root = tk.Tk()
root.title("WELCOME")
root.geometry("1000x600")
root.configure(background='grey')
path = "marcus-wallis-416821-unsplash(1).jpg"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(root, image = img)
panel.image=img
#The Pack geometry manager packs widgets in rows or columns.
panel.pack(side = "bottom", fill = "both", expand = "yes")


def for_start_button():
   cam = cv2.VideoCapture(1)

   cv2.namedWindow("test")

   img_counter = 0

   while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
   #sa()
   cam.release()
   
   cv2.destroyAllWindows()
   DIGIT_WIDTH = 10 
   DIGIT_HEIGHT = 20
   IMG_HEIGHT = 28
   IMG_WIDTH = 28
   CLASS_N = 10 # 0-9

#This method splits the input training image into small cells (of a single digit) and uses these cells as training data.
#The default training image (MNIST) is a 1000x1000 size image and each digit is of size 10x20. so we divide 1000/10 horizontally and 1000/20 vertically.
   def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

   def load_digits(fn):
    print('loading "%s for training" ...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
    resized_digits = []
    for digit in digits:
        resized_digits.append(imresize(digit,(IMG_WIDTH, IMG_HEIGHT)))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return np.array(resized_digits), labels

   def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img, 
                 orientations=10, 
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1), 
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

#define a custom model in a similar class wrapper with train and predict methods
   class KNN_MODEL():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

   class SVM_MODEL():
    def __init__(self, num_feats, C = 1, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF) #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1,self.features))
        return results[1].ravel()


   def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    final_bounding_rectangles = []
    #find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    
    for r,hr in zip(bounding_rectangles, hierarchy):
        x,y,w,h = r
        #this could vary depending on the image you are trying to predict
        #we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        #we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        #ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        #read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            final_bounding_rectangles.append(r)    
    print(final_bounding_rectangles)
    return final_bounding_rectangles


   def proc_user_img(img_file, model):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)    
    blank_image = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    blank_image.fill(255)

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
    #print(digits_rectangles)
    #s=digits_rectangles.len()
    #print(len(digits_rectangles),"sadassadasd")
    #print(digits_rectangles.count()/2)
    #w=0
    #im1=cv2.imread('8.png')
    for rect in digits_rectangles:
        x,y,w,h = rect
        #cv2.rectangle(im1,(x,y),(x+w+70,y+h+70),(0,255,0),cv2.FILLED)
        cv2.rectangle(im,(x,y),(x+w+30,y+h+30),(0,0,255),cv2.FILLED)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        im_digit = imresize(im_digit,(IMG_WIDTH ,IMG_HEIGHT))
        #w +=1 
        hog_img_data = pixels_to_hog_20([im_digit])  
        pred = model.predict(hog_img_data)
        #cv2.putText(im, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        #cv2.putText(blank_image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
    #cv2.rectangle(im1,(,),(x+w+50,y+h+50),(0,255,0),cv2.FILLED)
    plt.imshow(im)
    cv2.imwrite("q.png",im)
    img = cv2.imread('q.png')
    height,width = img.shape[:2]
    cv2.line(img,(0,int(height/2)),(int(width),int(height/2)),(255,0,0),3)
    cv2.line(img,(int(width/3),0),(int(width/3),int(height)),(255,0,0),3)
    cv2.line(img,(int((width*2)/3),0),(int((width*2)/3),int(height)),(255,0,0),3)
    cv2.imwrite("q.png",img)
    cv2.imwrite("final_digits.png",blank_image) 
    cv2.destroyAllWindows()   
    path="original_overlay.png"        
    #######################################################
    img = cv2.imread('q.png')
#Draw a diagonal blue line with thickness of 5
    height,width = img.shape[:2]
    cv2.line(img,(0,int(height/2)),(int(width),int(height/2)),(255,0,0),3)
    cv2.line(img,(int(width/3),0),(int(width/3),int(height)),(255,0,0),3)
    cv2.line(img,(int((width*2)/3),0),(int((width*2)/3),int(height)),(255,0,0),3)
    #cv2.imshow('sad',img)
    cv2.imwrite("original_overlay.png",img) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()  
    #print("sasa",w)
   def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  #row-wise ordering


#this function processes a custom training image
#see example : custom_train.digits.jpg
#if you want to use your own, it should be in a similar format
   def load_digits_custom(img_file):
    train_data = []
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
    
    #sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))
    
    for index,rect in enumerate(digits_rectangles):
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w+100,y+h+100),(0,255,0),2)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        
        im_digit = imresize(im_digit,(IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class%10)

        if index>0 and (index+1) % 10 == 0:
            start_class += 1
    cv2.imwrite("training_box_overlay.png",im)
    
    return np.array(train_data), np.array(train_target)

#------------------data preparation--------------------------------------------

   TRAIN_MNIST_IMG = 'digits.png' 
   TRAIN_USER_IMG = 'custom_train_digits.jpg'
   TEST_USER_IMG = img_name

#digits, labels = load_digits(TRAIN_MNIST_IMG) #original MNIST data (not good detection)
   digits, labels = load_digits_custom(TRAIN_USER_IMG) #my handwritten dataset (better than MNIST on my handwritten digits)

   print('train data shape',digits.shape)
   print('test data shape',labels.shape)

   digits, labels = shuffle(digits, labels, random_state=256)
   train_digits_data = pixels_to_hog_20(digits)
   X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

#------------------training and testing----------------------------------------

   model = KNN_MODEL(k = 3)
   model.train(X_train, y_train)
   preds = model.predict(X_test)
   print('Accuracy: ',accuracy_score(y_test, preds))

   model = KNN_MODEL(k = 4)
   model.train(train_digits_data, labels)
   proc_user_img(TEST_USER_IMG, model)



   model = SVM_MODEL(num_feats = train_digits_data.shape[1])
   model.train(X_train, y_train)
   preds = model.predict(X_test)
   print('Accuracy: ',accuracy_score(y_test, preds))

   model = SVM_MODEL(num_feats = train_digits_data.shape[1])
   model.train(train_digits_data, labels)
   proc_user_img(TEST_USER_IMG, model)

#------------------------------------------------------------------------------

#----------------------------------------------------
def check_status():
 t=1
 cam = cv2.VideoCapture(1)
#cv2.namedWindow("test")

 img_counter = 0

 while True:
    ret, frame = cam.read()
    #cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif t == 1:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        break
   #sa()
 cam.release()
   
 cv2.destroyAllWindows()
 rt = tk.Toplevel()
 #root.destroy()
 rt.title("WELCOME")
 rt.geometry("650x490")
 rt.configure(background='black')
 
 
 DIGIT_WIDTH = 10 
 DIGIT_HEIGHT = 20
 IMG_HEIGHT = 28
 IMG_WIDTH = 28
 CLASS_N = 10 # 0-9

#This method splits the input training image into small cells (of a single digit) and uses these cells as training data.
#The default training image (MNIST) is a 1000x1000 size image and each digit is of size 10x20. so we divide 1000/10 horizontally and 1000/20 vertically.
 def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

 def load_digits(fn):
    print('loading "%s for training" ...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
    resized_digits = []
    for digit in digits:
        resized_digits.append(imresize(digit,(IMG_WIDTH, IMG_HEIGHT)))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return np.array(resized_digits), labels

 def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img, 
                 orientations=10, 
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1), 
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

#define a custom model in a similar class wrapper with train and predict methods
 class KNN_MODEL():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

 class SVM_MODEL():
    def __init__(self, num_feats, C = 1, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF) #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1,self.features))
        return results[1].ravel()


 def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    final_bounding_rectangles = []
    #find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    
    for r,hr in zip(bounding_rectangles, hierarchy):
        x,y,w,h = r
        #this could vary depending on the image you are trying to predict
        #we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        #we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        #ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        #read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            final_bounding_rectangles.append(r)    

    return final_bounding_rectangles

 em=0
 def proc_user_img(img_file, model):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)    
    blank_image = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    blank_image.fill(255)

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    global em
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2 = cv2.imread('q.png')  
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
    em=len(get_digits(contours,hierarchy))
    print(em)
    #em=digits_rectangles
    for rect in digits_rectangles:
        x,y,w,h = rect
        cv2.rectangle(im2,(x,y),(x+w+50,y+h+50),(0,255,0),cv2.FILLED)
        cv2.rectangle(im2,(x-20,y-20),(x+w+50,y+h+50),(0,255,0),cv2.FILLED)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        im_digit = imresize(im_digit,(IMG_WIDTH ,IMG_HEIGHT))

        hog_img_data = pixels_to_hog_20([im_digit])  
        pred = model.predict(hog_img_data)
     #   cv2.putText(im, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    #    cv2.putText(blank_image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
    
    #t3.config(text="sadas")
    plt.imshow(im)
    cv2.imwrite("oo.png",im) 
    cv2.imwrite("final_digits.png",im2)
    img = cv2.imread('original_overlay.png')
    
    
    


 def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  #row-wise ordering


#this function processes a custom training image
#see example : custom_train.digits.jpg
#if you want to use your own, it should be in a similar format
 def load_digits_custom(img_file):
    train_data = []
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    #global em
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
    #em = get_digits(contours,hierarchy)
    #sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))
    
    for index,rect in enumerate(digits_rectangles):
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        
        im_digit = imresize(im_digit,(IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class%10)

        if index>0 and (index+1) % 10 == 0:
            start_class += 1
    cv2.imwrite("training_box_overlay.png",im)
    
    return np.array(train_data), np.array(train_target)

#------------------data preparation--------------------------------------------
 img_counter=0
 TRAIN_MNIST_IMG = 'digits.png' 
 TRAIN_USER_IMG = 'custom_train_digits.jpg'
 img_name = "opencv_frame_{}.png".format(img_counter)

 TEST_USER_IMG =img_name
 #TEST_USER_IMG ='original_overlay.png' 
 img_counter +=1

#digits, labels = load_digits(TRAIN_MNIST_IMG) #original MNIST data (not good detection)
 digits, labels = load_digits_custom(TRAIN_USER_IMG) #my handwritten dataset (better than MNIST on my handwritten digits)

 print('train data shape',digits.shape)
 print('test data shape',labels.shape)

 digits, labels = shuffle(digits, labels, random_state=256)
 train_digits_data = pixels_to_hog_20(digits)
 X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

#------------------training and testing----------------------------------------

 model = KNN_MODEL(k = 3)
 model.train(X_train, y_train)
 preds = model.predict(X_test)
 print('Accuracy: ',accuracy_score(y_test, preds))

 model = KNN_MODEL(k = 4)
 model.train(train_digits_data, labels)
 proc_user_img(TEST_USER_IMG, model)



 model = SVM_MODEL(num_feats = train_digits_data.shape[1])
 model.train(X_train, y_train)
 preds = model.predict(X_test)
 print('Accuracy: ',accuracy_score(y_test, preds))

 model = SVM_MODEL(num_feats = train_digits_data.shape[1])
 model.train(train_digits_data, labels)
 proc_user_img(TEST_USER_IMG, model)

 
 pa1 ='final_digits.png'

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
 im1 = ImageTk.PhotoImage(Image.open(pa1))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
 paa = tk.Label(rt, image = im1)
 
 paa.image=im1
#The Pack geometry manager packs widgets in rows or columns.
 paa.pack(side = "bottom", fill = "both", expand = "yes")
 
 n1=tk.Label(rt,text="1",font='Helvetica 10 bold')
 n1.place(x=70,y=0)
 n1.configure(width=10, height=2,font='Helvetica 10 bold')
 
 n2=tk.Label(rt,text="2")
 n2.place(x=300,y=0)
 n2.configure(width=10, height=2,font='Helvetica 10 bold')
 
 n3=tk.Label(rt,text="3")
 n3.place(x=530,y=0)
 n3.configure(width=10, height=2,font='Helvetica 10 bold')
 
 n4=tk.Label(rt,text="4")
 n4.place(x=70,y=430)
 n4.configure(width=10, height=2,font='Helvetica 10 bold')
 
 n5=tk.Label(rt,text="5")
 n5.place(x=300,y=430)
 n5.configure(width=10, height=2,font='Helvetica 10 bold')
 
 n6=tk.Label(rt,text="6")
 n6.place(x=530,y=430)
 n6.configure(width=10, height=2,font='Helvetica 10 bold')
 r="No. of Empty Slots :-"
 emp=(str(em))
 print(em)
 print(emp)
 emp1=r+emp
 var=StringVar()
 var.set(emp1)
 #n7=tk.Label(rt,textvariable=var)
 #n7.place(x=250,y=200)
 #n7.configure(width=30, height=2,font='Helvetica 10 bold')
 #n7.config(textvariable=em)
#--------------------------------------------------------

#-----------------------------------------------------
def quit():
 sys.exit()


#------------------------------------------------------
def inside():
 rot = tk.Toplevel(root)
 #root.destroy()
 rot.title("WELCOME")
 rot.geometry("1000x600")
 rot.configure(background='grey')

 pa = "609aafd56742982d26765f49599e78ed.jpg"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
 im = ImageTk.PhotoImage(Image.open(pa))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
 p = tk.Label(rot, image = img)

#The Pack geometry manager packs widgets in rows or columns.
 p.pack(side = "bottom", fill = "both", expand = "yes")
 p.image=pa

 
 l = tk.Label(rot, text="SMART PARKING SYSTEM",font='Helvetica 15 bold')

#The Pack geometry manager packs widgets in rows or columns.
#l.pack(side = "bottom", fill = "both", expand = "yes")
 l.place(x=60, y=0)
 l.configure(width=80, height=2)

 b1 =tkinter.Button(rot, text="Start",bg='grey',command=for_start_button)
 b1.place(relx=1, x=-10, y=220, anchor="ne")
 b1.config(width=20, height=2)
 
 b3 =tkinter.Button(rot, text="Exit",bg='grey',command=quit)
 b3.place(relx=1, x=-10, y=340, anchor="ne")
 b3.config(width=20, height=2)
 
 b2 =tk.Button(rot, text="Check Status",bg='grey',command=check_status)
 b2.place(relx=1, x=-10, y=280, anchor="ne")
 b2.config(width=20, height=2)
 	
#Start the GUI
#rot.mainloop()
def login():
 connection = pymysql.connect(host="localhost", user="root", passwd="", database="Sahil")
 cursor = connection.cursor()
 var1=text.get("1.0","end-1c")
 var2=t.get("1.0","end-1c")
 print(var2)
 cursor.execute("SELECT Phoneno from Users WHERE Username=%s AND Password=%s",(var1,var2))
 rows = cursor.fetchall()
 if rows:
    inside()
    
 else:
    showerror(title = "Error", message = "Wrong Credentials")
 
 connection.commit()
 connection.close()
 
def signup():
 w = tk.Toplevel(root)
 

 w.title("Registration")
 w.geometry("600x400")
 #w.configure(background='609aafd56742982d26765f49599e78ed.jpg')
 #pth = "609aafd56742982d26765f49599e78ed.jpg"
 #img = ImageTk.PhotoImage(Image.open(pth))
 #pl = tk.Label(w, image = img)
 #pl.pack(side = "top", fill = "both", expand = "yes") 
 def s1():
  connection = pymysql.connect(host="localhost", user="root", passwd="", database="Sahil")
  cursor = connection.cursor()

  var1=tu.get("1.0","end-1c")
  var2=tp.get("1.0","end-1c")
  var3=et.get("1.0","end-1c")
  var4=int(pht.get("1.0","end-1c")) 
  print (type(var4))
  
  cursor.execute("INSERT INTO Users VALUES (%s, %s, %s, %s)", (var1,var2,var3,var4,))
  retrive = "Select * from Users;"

  #executing the quires
  cursor.execute(retrive)
  rows = cursor.fetchall()
  for row in rows:
    print(row)
  connection.commit()
  connection.close()

#closing a window
 def close_window (): 
    w.destroy()
#---------------------------------------------------

 user1=tk.Label(w, text="Username",font='Helvetica 13 bold')
 user1.place(x=10,y=10)
 user1.configure(width=10, height=2)
 
 p=tk.Label(w, text="Password",font='Helvetica 13 bold')
 p.place(x=10,y=40)
 p.configure(width=10, height=2)

 tu = tk.Text(w)
 tu.place(x=130,y=10)
 tu.configure(width=35,height=2)

 tp = tk.Text(w)
 tp.place(x=130,y=40)
 tp.configure(width=35,height=2)
	
 em1=tk.Label(w, text="Email Id",font='Helvetica 13 bold')
 em1.place(x=10,y=70)
 em1.configure(width=10, height=2)
 
 ph=tk.Label(w, text="Phone No.",font='Helvetica 13 bold')
 ph.place(x=10,y=110)
 ph.configure(width=10, height=2)

 et = tk.Text(w)
 et.place(x=130,y=70)
 et.configure(width=35,height=2)

 pht = tk.Text(w)
 pht.place(x=130,y=110)
 pht.configure(width=35,height=2)

 sav=tkinter.Button(w, text="Save",bg='grey',command=s1)
 sav.place(x=70,y=200)
 sav.config(width=20,height=2)

 clos=tkinter.Button(w, text="Close",bg='grey',command=close_window)
 clos.place(x=240,y=200)
 clos.config(width=20,height=2)
  

 
#----------------------------------------------------------------------------
l = tk.Label(root, text="SMART PARKING SYSTEM",font='Helvetica 15 bold')

#The Pack geometry manager packs widgets in rows or columns.
#l.pack(side = "bottom", fill = "both", expand = "yes")
l.place(x=60, y=0)
l.configure(width=80, height=2)

###################################################################
#########Label
user=tk.Label(root, text="Username",font='Helvetica 12 bold')
user.place(x=300,y=200)
user.configure(width=10, height=2)

pas=tk.Label(root, text="Password",font='Helvetica 12 bold')
pas.place(x=300,y=270)
pas.configure(width=10, height=2)

text = tk.Text(root)
text.place(x=413,y=200)
text.configure(width=35,height=2)

t = tk.Text(root)
t.place(x=413,y=270)
t.configure(width=35,height=2)

lo =tkinter.Button(root, text="Login",bg='grey',command=login)
lo.place(x=270,y=350)
lo.config(width=20,height=2)

sign =tkinter.Button(root, text="Signup",bg='grey',command=signup)
sign.place(x=480,y=350)
sign.config(width=20,height=2)

#connection.commit()
#connection.close()

#b =tkinter.Button(root, text="Start",bg='grey')

#b.place(relx=1, x=-10, y=220, anchor="ne")
#b.config(width=20, height=2)
#b1 =tk.Button(root, text="Check Status",bg='grey')

##b1.place(relx=1, x=-10, y=280, anchor="ne")
#b1.config(width=20, height=2)
#b2 =tkinter.Button(root, text="Exit",bg='grey')

#b2.place(relx=1, x=-10, y=340, anchor="ne")
#b2.config(width=20, height=2)

#Start the GUI
root.mainloop()
