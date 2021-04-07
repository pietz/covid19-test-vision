import streamlit as st
from matcher import TestMatcher
from PIL import Image
import numpy as np
import cv2

size = 1280

config = {
    "name": "Boson",
    "img_path": "ref_1.jpg",
    "t_crop": [215, 700, 285, 770],
    "c_crop": [215, 785, 285, 855],
}

st.title("COVID-19 Testreader")
matcher = TestMatcher(config)

f = st.file_uploader("Select a photo")

if f is not None:
    img = Image.open(f).convert("RGB")
    #print(img.size)
    img = np.array(img)[..., ::-1]
    #print(img.shape)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[0], img.shape[1]
    if h > w:
        size2 = int(round(h / w * size))
        img = cv2.resize(img, (size, size2), interpolation=cv2.INTER_AREA)
    else:
        size2 = int(round(w / h * size))
        img = cv2.resize(img, (size2, size), interpolation=cv2.INTER_AREA)
    #print(img.shape)

    res, img = matcher.analyze(img)
    if res == "no-test":
        st.header("No test was found in the image")
        st.write("Try to bring the test closer to the camera.")
    elif res == "invalid":
        st.header("The test is invalid")
        st.write("No visible 'C' indicates that the test is unused or invalid.")
    elif res == "positive":
        st.header("The test is positive")
        st.write("Please seek medical attention.")
    else:
        st.header("The test is negative")
        st.write("We have no reason to believe that you're sick.")
    st.image(img[..., ::-1], use_column_width=True)
