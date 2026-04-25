#!/usr/bin/env bash

pip install -r requirements.txt

python manage.py collectstatic --noinput

mkdir -p model

wget "https://drive.google.com/uc?export=download&id=1oZzSuFrAKoOPE3PHUAwOM7JyUGRxZhw1" -O model/brain_model.h5