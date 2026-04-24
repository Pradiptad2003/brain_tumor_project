#!/usr/bin/env bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading model..."
wget "https://drive.google.com/uc?id=1oZzSuFrAKoOPE3PHUAwOM7JyUGRxZhw1" -O model/brain_model.h5

echo "Applying migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput
