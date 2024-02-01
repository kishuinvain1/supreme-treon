import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64


def load_image():
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image)
       
    return path, opencv_image
       


	


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return np.array(image)
	

	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    #img = cv2.imread(saved_image)
    #img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    img = saved_image
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-15)

    color = (0,255,0)
        
        
    img = cv2.rectangle(img, start_pnt, end_pnt, color, 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 3, cv2.LINE_AA)	
    return img
        
    


def predict(model, url):
    return model.predict(url, confidence=50, overlap=70).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('Part Inspection')
    
    rf = Roboflow(api_key="ztkZZ5Bwjux1hvHq6IFW")
    project = rf.workspace("viswamanav-r-s-jwwu4").project("ok_golden_sample")
    model = project.version("2").model
        
    image, svd_img = load_image()

    result = st.button('Detect')
    if result:
        results = predict(model, svd_img)
        #results = predict(model2, url)
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No Object Detected")
        else:
            for i in range(len(results['predictions'])):
                new_img_pth = results['predictions'][i]['image_path']
                x = results['predictions'][i]['x']
                y = results['predictions'][i]['y']
                w = results['predictions'][i]['width']
                h = results['predictions'][i]['height']
                cl = results['predictions'][i]['class']
                cnf = results['predictions'][i]['confidence']

                svd_img = drawBoundingBox(svd_img,x, y, w, h, cl, cnf)

            st.write('DETECTION RESULTS')    
            st.image(svd_img, caption='Resulting Image')
           

if __name__ == '__main__':
    main()
