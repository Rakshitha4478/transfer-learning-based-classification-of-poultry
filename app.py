from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/poultry_model.h5")
classes = ['chicken', 'duck', 'turkey']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        file = request.files["image"]
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        prediction = classes[np.argmax(result)]

        return render_template("index.html", prediction=prediction, image=file.filename)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
