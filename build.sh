#!/usr/bin/env bash

pip install -r requirements.txt

python manage.py collectstatic --noinput

# Create model folder
mkdir -p model

# Download TFLite model
wget -O model/brain_model.tflite "https://drive.google.com/uc?export=download&id=1WX5XO7h6nLmgBqR9o_JdwT3j_y-stLyw"