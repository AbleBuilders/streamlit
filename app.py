
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

URI = "http://127.0.0.1/5000"

st.title("Neural Network Web App")
st.sidebar.markdown("## Input Images")

if st.button("Get some predictions"):
    response = requests.post(URI, data = {})
    reponse = json.loads(response.text)
    preds = response.get("prediction")
    image = response.get("image")
    image = np.reshape(image, (28, 28))
    
    st.image(image, width = 150)
