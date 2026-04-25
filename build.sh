#!/usr/bin/env bash

pip install -r requirements.txt

python manage.py collectstatic --noinput

# create model folder
mkdir -p model

# install gdown (extra safe)
pip install gdown

# download model (FINAL FIX)
gdown --id 1oZzSuFrAKoOPE3PHUAwOM7JyUGRxZhw1 -O model/brain_model.h5

# check file exists (IMPORTANT DEBUG)
ls model/