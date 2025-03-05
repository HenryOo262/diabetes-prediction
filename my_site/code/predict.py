import pandas
import numpy
import os
import joblib

def predict(df, loc, model='original'):

    if(model == 'original'):
        mscaler_path = os.path.join(loc, 'mscaler.joblib')
        model_path = os.path.join(loc, 'model.joblib')
        regre_path = os.path.join(loc, 'regre.joblib')
    elif(model == 'new'):
        mscaler_path = os.path.join(loc, 'mscaler.joblib')
        model_path = os.path.join(loc, 'user_trained_model1.joblib')
        regre_path = os.path.join(loc, 'user_trained_model2.joblib')
        if(not os.path.isfile(model_path) or not os.path.isfile(regre_path)):
            return None
        
    # gaussian nb model
    model = joblib.load(model_path)
    # logistic reg model
    regre = joblib.load(regre_path)

    if(os.path.isfile(mscaler_path)):
        # minmax scaler joblib file will only exists if you have trained normalized data, if you 
        # haven't trained normalized data the file wont exist and no need to normalize inputs
        mscaler = joblib.load(mscaler_path)
        df = mscaler.transform(df)

    nb = model.predict(df)
    lr = regre.predict(df)

    result = {
        'GaussianNB': 'Diabetes' if nb[0] == 1 else 'No Diabetes',
        'LogisticRegression': 'Diabetes' if lr[0] == 1 else 'No Diabetes',
    }

    return result