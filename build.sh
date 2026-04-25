#!/usr/bin/env bash

pip install -r requirements.txt

python manage.py collectstatic --noinput

mkdir -p model

# fix large file download (IMPORTANT)
gdown --id 1oZzSuFrAKoOPE3PHUAwOM7JyUGRxZhw1 -O model/brain_model.h5