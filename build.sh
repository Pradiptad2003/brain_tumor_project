#!/usr/bin/env bash

pip install -r requirements.txt

python manage.py collectstatic --noinput

mkdir -p model

pip install gdown

# FORCE download with fuzzy (important)
gdown --fuzzy "https://drive.google.com/file/d/1oZzSuFrAKoOPE3PHUAwOM7JyUGRxZhw1/view" -O model/brain_model.h5

ls model/