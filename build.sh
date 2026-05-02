#!/usr/bin/env bash

pip install -r requirements.txt

python manage.py collectstatic --noinput

mkdir -p model

