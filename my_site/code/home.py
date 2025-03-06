from flask import Blueprint, make_response, request, redirect, render_template, url_for
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from threading import Thread
import time
import shutil

import joblib
import numpy
import pandas
import uuid
import os
from my_site.code.predict import predict
from my_site.code.train import train_model
from my_site.code.file_validation import file_validation

pageViews = Blueprint('pageViews', __name__)

def call_route():
    while True:
        try:
            #response = requests.get(url_for('pageViews.ping'))  # Call the Blueprint route
            #print(response.json())  # Print response
            loc = os.path.join(os.path.dirname(__file__), 'saves')
            contents = os.listdir(loc)
            for content in contents:
                content_path = os.path.join(loc, content)
                if(os.path.isfile(content_path)):
                    os.remove(content_path)
                elif(os.path.isdir(content_path)):
                    shutil.rmtree(content_path)
                print(content_path)
        except Exception as e:
            print("Error:", e)
        time.sleep(5)  # Wait before the next call - a day

# Start the thread when the blueprint is imported
thread = Thread(target=call_route, daemon=True)
thread.start()

def savefolder():
    current = os.path.dirname(__file__)
    savefolder = os.path.join(current, 'saves')
    return savefolder

@pageViews.route('/')
def home():
    return redirect(url_for('pageViews.diabetes'))

@pageViews.route('/diabetes')
def diabetes():
    loadloc = os.path.join(os.path.dirname(__file__), '..', 'trained', 'describe.joblib')
    dsb = joblib.load(loadloc)
    pdimean = dsb['DiabetesPedigreeFunction'][1]
    resp = make_response(render_template('page.html', result=None))
    sessID = request.cookies.get('sessionID')
    # if not 
    # create folder and sends cookie
    if(not sessID):
        # or not os.path.isdir(os.path.join(savefolder() ,str(sessID)))
        sessID = str(uuid.uuid4())
        os.mkdir(os.path.join(os.path.dirname(__file__), 'saves', str(sessID)))
        resp.set_cookie('sessionID', sessID, max_age=10800)
    # if path
    else:
        if(not os.path.isdir(os.path.join(os.path.dirname(__file__), 'saves', str(sessID)))):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'saves', str(sessID)))
        details = None
        loadloc_details = os.path.join(savefolder(), str(sessID), 'details.joblib')
        if(os.path.isfile(loadloc_details)):
            details = joblib.load(loadloc_details)
        return render_template('page.html', details=details, pdimean=pdimean)
    return resp

@pageViews.route('/uploaded_dataset')
def uploaded_dataset():
    sessID = request.cookies.get('sessionID')
    loadloc_df = os.path.join(savefolder(), str(sessID), 'dataframe.joblib')
    if(os.path.isfile(loadloc_df)):
        df = joblib.load(loadloc_df)
        html = df.to_html()
        return html
    else:
        return '<h1> Please upload a dataset first !</h1>'
    
@pageViews.route('/x_train_describe')
def x_train_describe():
    sessID = request.cookies.get('sessionID')
    loadloc_df = os.path.join(savefolder(), str(sessID), 'describe.joblib')
    if(os.path.isfile(loadloc_df)):
        df = joblib.load(loadloc_df)
        html = df.to_html()
        return html
    else:
        return '<h1> Please train test split the uploaded dataset first ! </h1>'
    
@pageViews.route('/upload', methods=['POST'])
def upload():
    file = request.files['dataset']
    if 'dataset' not in request.files:
        return 'Error: Did not receive file', 400
    flag, message = file_validation(file=file)
    if not flag:
        return message, 400
    else:
        # reset file pointer
        file.seek(0) 
        df = pandas.read_csv(file)
        df.fillna(df.mean(), inplace=True)
        # save
        saveloc = os.path.join(savefolder(), str(request.cookies.get('sessionID')))
        saveloc_dataframe = os.path.join(saveloc, 'dataframe.joblib')
        # remove everythin inside before saving new dataset
        for filename in os.listdir(saveloc): # os.listdir returns list of filenames
            del_file = os.path.join(saveloc, filename)
            if os.path.isfile(del_file):
                os.remove(del_file)
        joblib.dump(df, saveloc_dataframe)
        # to html
        return redirect(url_for('pageViews.home'))
    
@pageViews.route('/preprocess')
def preprocess():
    type = request.args.get('type')
    mmscaler = MinMaxScaler().set_output(transform='pandas')
    # save
    loc_x_train = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'x_train.joblib')
    loc_x_test = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'x_test.joblib')
    loc_describe = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'describe.joblib')
    if(not os.path.isfile(loc_x_train) or not os.path.isfile(loc_x_test) or not os.path.isfile(loc_describe)):
        return 'Error: Please train test split a dataset first'
    x_train = joblib.load(loc_x_train)
    x_test = joblib.load(loc_x_test)
    x_train = mmscaler.fit_transform(x_train)
    x_test = mmscaler.transform(x_test)
    joblib.dump(x_train, loc_x_train)
    joblib.dump(x_test, loc_x_test)
    joblib.dump(x_train.describe(), loc_describe)
    joblib.dump(mmscaler, os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'mscaler.joblib'))
    return redirect(url_for('pageViews.home'))

@pageViews.route('/diabetes-predict', methods=['GET', 'POST'])
def diabetes_predict():
    model      = request.form.get('model', default='original', type=str)
    pregnancy  = int(request.form.get('pregnancy'))
    glucose    = int(request.form.get('glucose'))
    bp         = int(request.form.get('bp'))
    skinthickness = int(request.form.get('skinthickness'))
    insulin    = int(request.form.get('insulin'))
    bmi      = float(request.form.get('bmi'))
    pedigree = float(request.form.get('pedigree'))
    age      = int(request.form.get('age'))

    #loadloc = os.path.join(savefolder(), 'user_trained_describe.joblib')
    #dsb = joblib.load(loadloc)

    if model=='original':
        loadloc = os.path.join(os.path.dirname(__file__), '..', 'trained', 'describe.joblib')
        dsb = joblib.load(loadloc)
    elif model=='new':
        loadloc = os.path.join(savefolder(), str(request.cookies.get('sessionID')),'refxtrain.joblib')
        dsb = joblib.load(loadloc).describe()
    else:
        return 'Error: invalid choice'
        
    # clip
    pregnancy   = numpy.clip(pregnancy, dsb['Pregnancies'][3], dsb['Pregnancies'][7])
    glucose     = numpy.clip(glucose, dsb['Glucose'][3], dsb['Glucose'][7])
    bp          = numpy.clip(bp, dsb['BloodPressure'][3], dsb['BloodPressure'][7])
    skinthickness = numpy.clip(skinthickness, dsb['SkinThickness'][3], dsb['SkinThickness'][7])
    insulin     = numpy.clip(insulin, dsb['Insulin'][3], dsb['Insulin'][7])
    pedigree    = numpy.clip(pedigree, dsb['DiabetesPedigreeFunction'][3], dsb['DiabetesPedigreeFunction'][7])
    age         = numpy.clip(age, dsb['Age'][3], dsb['Age'][7])
    bmi         = numpy.clip(bmi, dsb['BMI'][3], dsb['BMI'][7])

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

    if model=='original':
        passloc = os.path.join(os.path.dirname(__file__), '..', 'trained')
    elif model=='new':
        passloc = os.path.join(savefolder(), str(request.cookies.get('sessionID')))
    result = predict(df, model=model, loc=passloc)

    if not result and model == 'new':
        return 'Error: You have not trained any model yet', 500
    return render_template('result.html', result=result)

@pageViews.route('/train')
def train():
    x_train_loc = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'x_train.joblib')
    y_train_loc = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'y_train.joblib')
    x_test_loc = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'x_test.joblib')
    y_test_loc = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'y_test.joblib')
    if(not os.path.isfile(x_train_loc) or not os.path.isfile(y_train_loc) or not os.path.isfile(x_test_loc) or not os.path.isfile(y_test_loc)):
        return 'Error: Please train test split the uploaded dataset first', 500
    x_train = joblib.load(x_train_loc)
    y_train = joblib.load(y_train_loc)
    x_test = joblib.load(x_test_loc)
    y_test = joblib.load(y_test_loc)
    result, message = train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, sessID=request.cookies.get('sessionID'))
    if not result:
        return message, 500
    return redirect(url_for('pageViews.home'))

@pageViews.route('/trainset')
def trainset():
    sessID = request.cookies.get('sessionID')
    x_train = os.path.join(savefolder(), str(sessID), 'x_train.joblib')
    y_train = os.path.join(savefolder(), str(sessID), 'y_train.joblib')
    if(os.path.isfile(x_train) and os.path.isfile(y_train)):
        xdf = joblib.load(x_train)
        ydf = joblib.load(y_train)
        df = pandas.concat([xdf, ydf], axis=1)
        html = df.to_html()
        return html
    else:
        return '<h1> Please train test split the uploaded dataset first ! </h1>'
    
@pageViews.route('/testset')
def testset():
    sessID = request.cookies.get('sessionID')
    x_test = os.path.join(savefolder(), str(sessID), 'x_test.joblib')
    y_test = os.path.join(savefolder(), str(sessID), 'y_test.joblib')
    if(os.path.isfile(x_test) and os.path.isfile(y_test)):
        xdf = joblib.load(x_test)
        ydf = joblib.load(y_test)
        df = pandas.concat([xdf, ydf], axis=1)
        html = df.to_html()
        return html
    else:
        return '<h1> Please train test split the uploaded dataset first ! </h1>'

@pageViews.route('/traintestsplit')
def traintestsplit():
    train = 0.8
    test = 0.2
    
    load_loc = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'dataframe.joblib')
    if(not os.path.isfile(load_loc)):
        return 'Error: Please upload a dataset first', 400
    df = joblib.load(load_loc)

    # features and label
    y = df[['Outcome']]
    x = df.drop(columns=['Outcome'])

    # split train test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train)

    save_loc = os.path.join(savefolder(), str(request.cookies.get('sessionID')))
    joblib.dump(x_train, os.path.join(save_loc, 'x_train.joblib'))
    joblib.dump(y_train, os.path.join(save_loc, 'y_train.joblib'))
    joblib.dump(x_test, os.path.join(save_loc, 'x_test.joblib'))
    joblib.dump(y_test, os.path.join(save_loc, 'y_test.joblib'))

    # reference frame is copy of xtrain dataframe which will be kept unedited for future references
    saveloc_refxtrain = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'refxtrain.joblib')
    joblib.dump(x_train.copy(), saveloc_refxtrain)
    saveloc_describe = os.path.join(savefolder(), str(request.cookies.get('sessionID')), 'describe.joblib')
    joblib.dump(x_train.describe(), saveloc_describe)

    return redirect(url_for('pageViews.home'))
















'''
@pageViews.route('/getHTML', methods=['POST'])
def getHTML():
    file = request.files['dataset']
    if 'dataset' not in request.files:
        return jsonify({'message': 'Error: Did not receive file'}), 400
    code, message = file_validation(file=file)
    if not code:
        return jsonify({'message': message}), 400
    else:
        # reset file pointer
        file.seek(0) 
        df = pandas.read_csv(file)
        # save
        saveloc = os.path.join(savefolder(), str(request.cookies.get('sessionID')))
        joblib.dump(df, os.path.join(saveloc, 'dataset.joblib'))
        # to html
        html = df.to_html()
        return html, 200
'''