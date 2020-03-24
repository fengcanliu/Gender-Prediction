import os

from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import static.predict as predict
import static.train_model as train
import numpy as np


flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Name Gender predictor",
		  description = "Predict results using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')


model = app.model('Prediction params', 
				  {'name': fields.String(required = True,
				  							   description="Name",
    					  				 	   help="Name cannot be blank"),
				  'method': fields.String(required = True,
				  							description="Method",
    					  				 	help="Method cannot be blank")})

if not (os.path.isfile('./models/decisiontreemodel.pkl') or (os.path.isfile('./models/dicmodel.pkl'))):
	train.features = np.vectorize(train.features)
	dv, clf = train.vectorizer((train.preprocessing('./models/name_gender.csv')))


if not (os.path.isfile('./static/mnb_model.pkl') or (os.path.isfile('./models/cv_model.pkl'))):
	cv, clf = train.mnb(train.preprocessing('./models/name_gender.csv'))


clf = joblib.load('./models/mnb_model.pkl')
cv = joblib.load('./models/cv_model.pkl')
dclf = joblib.load('./models/decisiontreemodel.pkl')
dv = joblib.load('./models/dicmodel.pkl')
train.features = np.vectorize(train.features)


@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			print(formData)
			# data = [val for val in formData.values()]
			# prediction = classifier.predict(data)
			name = formData['name']
			method = formData['method']

			if (method == "MNB"):
				gender = predict.gender_predictor_mnb(name, cv, clf)
			else:
				gender = predict.gender_predictor_dt(name, dv, dclf)

			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Prediction: " + str(gender)
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			print(str(error))
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})