import pandas
import numpy
import os
import joblib

def predict(df, model='original'):

    if(model == 'original'):
        current = os.path.dirname(__file__)
        mscaler_path = os.path.join(current, 'mscaler.joblib')
        model_path = os.path.join(current, 'model.joblib')
        regre_path = os.path.join(current, 'regre.joblib')
    elif(model == 'new'):
        path = os.path.dirname(__file__)
        path = os.path.join(path, '..', 'code', 'saves')
        mscaler_path = os.path.join(path, 'user_trained_mscaler.joblib')
        model_path = os.path.join(path, 'user_trained_model1.joblib')
        regre_path = os.path.join(path, 'user_trained_model2.joblib')
        if(not os.path.isfile(mscaler_path) or not os.path.isfile(model_path) or not os.path.isfile(regre_path)):
            return None

    # minmax scaler
    mscaler = joblib.load(mscaler_path)
    # gaussian nb model
    model = joblib.load(model_path)
    # logistic reg model
    regre = joblib.load(regre_path)

    df = mscaler.transform(df)

    nb = model.predict(df)
    lr = regre.predict(df)

    result = {
        'GaussianNB': 'Diabetes' if nb[0] == 1 else 'No Diabetes',
        'LogisticRegression': 'Diabetes' if lr[0] == 1 else 'No Diabetes',
    }

    return result