#!bin/bash


docker build -t gender_predictor_ui .

docker run -p 3000:3000 gender_predictor_ui
