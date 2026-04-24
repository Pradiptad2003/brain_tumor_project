import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "brain_model.h5")

model = load_model(model_path)

IMG_SIZE = 150

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def index(request):
    result = None
    error = None

    if request.method == "POST":
        if 'image' not in request.FILES:
            error = "Please select an image"
            return render(request, "index.html", {"error": error})

        image = request.FILES['image']

        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_path = fs.path(filename)

        img = preprocess_image(file_path)
        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            result = "🧠 Tumor Detected (YES)"
        else:
            result = "😊 No Tumor (NO)"

    return render(request, "index.html", {"result": result, "error": error})