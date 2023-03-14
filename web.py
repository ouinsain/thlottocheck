# Import all necessary packages
import streamlit as st
import pandas as pandas
import numpy as numpy
import urllib.request
# import fastai
# from fastai.learner import load_learner
# from fastai.vision.all import PILImage
import glob
from random import shuffle
import requests
import datetime

from PIL import Image
import pytesseract
import cv2
import numpy as np
import imutils
# from imutils.contours import sort_contours
import re
import pandas as pd
import json
from pandas import json_normalize
import ast

# Create simple title
st.title('Lottery Tickets Check')

# Sidebar
st.sidebar.write('### Select lottery image to check')
# Add radio button
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])

# Load images from validate set & shuffle
valid_imgs = glob.glob('images/lottery/*/*')
# shuffle(valid_imgs)

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_imgs)
else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                    type=['png','jpg','jpeg'],
                                    accept_multiple_files=False)
    if fname is None:
        fname = valid_imgs[0]

# Sidebar enter date
lott_date = st.date_input("Select lottery date (yyyy-mm-dd) ", datetime.date(2023, 1, 31))
st.write("Lottery date ", lott_date)
# print(type(lott_date))

# Define predict function
def predict(img, learn):

    # Predict from model
    pred, pred_idx, pred_prob = learn.predict(img)

    # Display result
    st.success(f"This is {pred} cookie with probability of {pred_prob[pred_idx]*100:.02f}%")

    st.image(img, use_column_width=True)

# Define API call to GLO website
def call_api(date, lottery_no):
    # payload = st.json({"number": [{"lottery_num": lottery_no}],"period_date": date})
    payload = {"number": [{"lottery_num": lottery_no}],"period_date": date}
    # print(date)
    # print(type(date))

    url = "https://www.glo.or.th/api/checking/getcheckLotteryResult"
    
    resp = requests.post(url, json=payload)

    # st.json(resp.content.decode())

    conv_resp = resp.json()
    # print(conv_resp['response']['result'][0]['status_data'])
    st.write("Number : ", lottery_no)
    st.write("Reward : ", "ไม่ถูกรางวัล" if conv_resp['response']['result'][0]['status_data'] == [] else conv_resp['response']['result'][0]['status_data'] )
    st.markdown("""---""")
    # return response.content.decode()

def convert_gray(img):

    # Display original image
    st.image(img, width=400)

    # Check image size and resize if too large image
    cv2img = np.array(img)
    height, width, channels = cv2img.shape
    scale_pct = 60
    width_resize = int(cv2img.shape[1]*scale_pct / 100)
    height_resize = int(cv2img.shape[0]*scale_pct / 100)
    dim = (width_resize, height_resize)

    if (width > 900 and height > 900) :
        img_resize = cv2.resize(cv2img, dim, interpolation = cv2.INTER_AREA)
    else:
        img_resize = cv2img

    # Convert to grayscale
    img_gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    (H, W) = img_gray.shape

    # st.image(img_gray, use_column_width=True)

    return img_gray

def edge_detect(img, kernel_blur, kernel_thresh, kernel_morph):

    # smooth the image using a 3x3 Gaussian blur and then apply a
    # blackhat morpholigical operator to find dark regions on a light background
    img_gray = cv2.GaussianBlur(img, (15, 15), 0)
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_blur)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel_thresh)
    thresh = cv2.threshold(gradX,   90, 255, cv2.THRESH_BINARY)[1]

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_morph)
    thresh_erode = cv2.erode(thresh_close, None, iterations=3)

    # st.image(img, use_column_width=True)
    # st.image(gradX, use_column_width=True)
    # st.image(thresh, use_column_width=True)
    # st.image(thresh_close, use_column_width=True)
    # st.image(thresh_erode, use_column_width=True)

    return thresh_erode

def contour(img, orig_img, kernel_morph):

    # find contours in the thresholded image
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize the bounding box associated with the MRZ
    box = np.empty((0, 4), dtype=int)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then derive the
        # how much of the image the bounding box occupies in terms of
        # both width and height
        (x, y, w, h) = cv2.boundingRect(c)
        print("x:",x,"y:",y,"w:",w,"h:",h, "h/w: ", h/w)
        # st.image(np.array(img[y:y + h, x:x + w]), width=200)
        # percentWidth = w/float(W)
        # percentHeight = h/float(H)

        # Only detected area that is not square and width > height should be area of ticket number
        if h/w < 0.228 and w < 350 :
            box = np.append(box, np.array([(x, y, w, h)]), axis=0)
            # st.image(np.array(img[y:y + h, x:x + w]), width=200)
    
    # for i in box:
    #     # Assign region of interest in lottery images
    #     roi = img_resize[ i[1]:i[1]+i[3], i[0]:i[0]+i[2] ]

    mrz = []
    cnts = len(box)
    print(cnts)
    round = 0

    # pad the bounding box since we applied erosions and now need to
    # re-grow it
    for x, y, w, h in box:
        print(x,y,w,h,"\n")
        pX = int((x + w) * 0.022)
        # High border
        pY = int((y + h) * 0.021)
        # Lower border
        pH = int((y + h) * 0.021)
        # print(pX, pY,"\n")
        if x != 0 and y != 0:
            (x, y) = (x - pX, y - pY)
        # (w, h) = (w + (pX * 2), h + (pY * 2))
        (w, h) = (w + (pX * 2), h + pH)
        print(x,y,w,h)

        # # extract the padded MRZ from the image and convert it to gray
        if round <= cnts:
            img_gray = cv2.cvtColor(np.array(orig_img[y:y + h, x:x + w]), cv2.COLOR_BGR2GRAY)
            # mrz.append(cv2.cvtColor(np.array(orig_img[y:y + h, x:x + w]), cv2.COLOR_BGR2GRAY))
            round = round+1
            print("round:", round)

            # Apply threshold
            blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel_morph)
            thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_OTSU)[1]
            erode = cv2.erode(thresh, None, iterations=1)

            # Apply negation
            inv = 255 - erode
            mrz.append(inv)
            # st.image(inv, use_column_width=True)

    return mrz

def img_ocr(img):

    custom_config = r'--oem 3 --psm 6 outputbase digits'
    lottery_no = []

    for cnts in img:
        # OCR image
        # st.image(cnts, width=200)
        data = pytesseract.image_to_string(cnts, config=custom_config)
        # Filter only number
        data = re.sub("[^0-9]", "", data)
        if(len(data.strip('')) == 6):
            lottery_no.append(data)
            st.image(cnts, width=200)
            # st.write("Number - ", data)
    
    return lottery_no

# Open image from selected radio button
# img = PILImage.create(fname)
img = Image.open(fname)

# # Call predict func
# predict(img, learn_inf)

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 15))
rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (37, 25))
rectKernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (39, 15))

# Convert to grey image
img_gray = convert_gray(img)

if st.button('Check'):
    # Egge detection
    img_mrz = edge_detect(img_gray, rectKernel, rectKernel2, rectKernel3)

    # Finding contour
    cv2img = np.array(img)
    img_contour = contour(img_mrz, cv2img, rectKernel)

    # OCR lottery image
    lotto_no = img_ocr(img_contour)

    for i in lotto_no :
        resp_api = call_api(lott_date.strftime("%Y-%m-%d"), i)