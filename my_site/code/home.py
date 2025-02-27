from flask import Blueprint, session, request, flash, redirect, render_template, url_for
from flask_mysqldb import MySQL
from pprint import pp as p
from functools import wraps

import joblib
import numpy
import pandas
from my_site.trained.predict import predict
from my_site.code.train import train_model

pageViews = Blueprint('pageViews', __name__)

@pageViews.route('/')
def asthma():
    return redirect(url_for('pageViews.diabetes'))

@pageViews.route('/diabetes')
def diabetes():
    return render_template('page.html', result=None)

@pageViews.route('/diabetes-predict', methods=['GET', 'POST'])
def diabetes_predict():
    model      = request.form.get('model', default='original', type=str)
    pregnancy  = int(request.form.get('pregnancy'))
    glucose    = int(request.form.get('glucose'))
    bp         = int(request.form.get('bp'))
    skinthickness = int(request.form.get('skinthickness'))
    insulin    = float(request.form.get('insulin'))
    bmi      = str(request.form.get('bmi'))
    pedigree = float(request.form.get('pedigree'))
    age      = int(request.form.get('age'))

    age = numpy.clip(age, 21, 81)

    df = pandas.DataFrame({
        'Pregnancies': [pregnancy],
        'Glucose': [glucose],
        'BloodPressure': [bp], 
        'SkinThickness': [skinthickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [pedigree],
        'Age': [age],
    })

    result = predict(df, model=model)

    if not result and model == 'new':
        return 'Error: You have not trained any model yet', 500
    return render_template('result.html', result=result)

@pageViews.route('/train', methods=['POST'])
def train():
    if 'dataset' not in request.files:
        return 'Error: Cannot receive file', 400

    file = request.files['dataset']

    result = train_model(file)

    if not result:
        return 'Error: Missing columns or mismatch datatypes', 500
    return render_template('trained_result.html', result=result)