
import json
import random
import tensorflow as tf
import numpy as np
from flask import Flask, requests

app = Flask(__name__)

(_,_), (x_test, _) = tf.keras.datasets.mnist.load_data()
test= test/255

model = tf.keras.models.load_model("model.h5")
feature_model = tf.keras.models.Model(
    model.input,
    [layer.output for layer in model.layers]
)

def get_prediction():
    index = np.random.choice(x_test.shape[0])


@app.route("/")
def index():
    return "something."

if __name__ == "__main__":
    app.run()
