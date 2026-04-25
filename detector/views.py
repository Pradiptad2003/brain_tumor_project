import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tflite_runtime.interpreter as tflite

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "brain_model.tflite")

interpreter = None

# SAFE LOAD (no crash)
if os.path.exists(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("✅ TFLite Model Loaded")
else:
    print("❌ Model NOT found")

IMG_SIZE = 150

def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("Image not readable")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def index(request):
    result = None
    error = None

    if interpreter is None:
        return render(request, "index.html", {
            "error": "Model not loaded on server"
        })

    if request.method == "POST":
        if 'image' not in request.FILES:
            return render(request, "index.html", {
                "error": "Please upload image"
            })

        image = request.FILES['image']

        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_path = fs.path(filename)

        img = preprocess_image(file_path)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if prediction > 0.5:
            result = "🧠 Tumor Detected"
        else:
            result = "😊 No Tumor"

    return render(request, "index.html", {
        "result": result,
        "error": error
    })