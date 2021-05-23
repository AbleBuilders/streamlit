
import json
import random
import tensorflow as tf
import numpy as np
from flask import Flask, request

app = Flask(__name__)

(_,_), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test= x_test/255

model = tf.keras.models.load_model("model.h5")
feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        preds, image = get_prediction()
        final_pred = [p.tolist() for p in preds]
        return json.dumps({
            "prediction" : final_pred,
            "image" : image.tolist()
        })
    return "something."

if __name__ == "__main__":
    app.run()
