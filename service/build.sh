#!bin/bash

docker build -t gender_predictor .

docker run -it -p 5000:5000 gender_predictor
