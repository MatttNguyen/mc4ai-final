import streamlit as st
import pandas as pd
import cv2
from deepface import DeepFace
import numpy as np
from datetime import datetime
import os
from time import process_time

df = pd.read_csv("timelog.csv")
st.title("Image Processing")

tab1, tab2, tab3 = st.tabs(["Face Register", "Face Verify", "Time Log"])

with tab1:
    name = st.text_input("Input Name")
    img = st.camera_input("Take a picture", 0)
    if img is not None:
        bytes_data = img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if name == "":
            st.error("Please input name")
        else:
            cv2.imwrite("faces/" + name + ".jpg", cv2_img)
            st.success("Register Done")

with tab2:
    if os.path.exists("faces/representations_vgg_face.pkl"):
        os.remove("faces/representations_vgg_face.pkl")
    img_2 = st.camera_input("Take a picture", 1)
    if img_2 is not None:
        bytes_data_2 = img_2.getvalue()
        cv2_img_2 = cv2.imdecode(np.frombuffer(bytes_data_2, np.uint8), cv2.IMREAD_COLOR)
        st.spinner("Please wait...")
        t1 = process_time()
        verified = DeepFace.find(cv2_img_2, "faces/")
        t2 = process_time()
        name_verified = verified[0]["identity"][0][7:-4]
        accuracy = int((1 - verified[0]["VGG-Face_cosine"][0]) * 100)
        st.success(name_verified + ": " + str(accuracy) + "%")
        st.write("Process time: " + str(round(t2 - t1, 3)) + "s")
        new_row = {"Name" : name_verified, "Time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}
        df= pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("timelog.csv", index=False)

with tab3:
    st.dataframe(df)

