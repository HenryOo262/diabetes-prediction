import pandas
import os
import numpy
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib

def train_model(x_train, y_train, x_test, y_test, sessID):
    # minmax scaler
    # mscaler = MinMaxScaler().set_output(transform='pandas')

    # model
    model1 = GaussianNB()
    model2 = LogisticRegression()

    # train
    model1 = model1.fit(x_train, y_train)
    model2 = model2.fit(x_train, y_train)

    saveloc = os.path.join(os.path.dirname(__file__), 'saves', sessID)

    joblib.dump(model1, os.path.join(saveloc, 'user_trained_model1.joblib'))
    joblib.dump(model2, os.path.join(saveloc, 'user_trained_model2.joblib'))
    #joblib.dump(mscaler, os.path.join(saveloc, 'user_trained_mscaler.joblib'))

    pred1 = model1.predict(x_test)
    pred2 = model2.predict(x_test)

    accuracy1 = accuracy_score(pred1, y_test)
    accuracy2 = accuracy_score(pred2, y_test)

    precision1 = precision_score(pred1, y_test)
    precision2 = precision_score(pred2, y_test)

    recall1 = recall_score(pred1, y_test)
    recall2 = recall_score(pred2, y_test)

    confusion_matrix1 = confusion_matrix(pred1, y_test, labels=[1, 0])
    confusion_matrix2 = confusion_matrix(pred2, y_test, labels=[1, 0])

    details = {
        'GaussianNB': {
            'Accuracy': accuracy1,
            'Precision': precision1,
            'Recall': recall1,
            'Confusion Matrix': confusion_matrix1,
            'Confusion Matrix Labels': ['Diabetes', 'No Diabetes'],
        },
        'LogisticRegression': {
            'Accuracy': accuracy2,
            'Precision': precision2,
            'Recall': recall2,
            'Confusion Matrix': confusion_matrix2,
            'Confusion Matrix Labels': ['Diabetes', 'No Diabetes'],
        }
    }

    joblib.dump(details, os.path.join(saveloc, 'details.joblib'))

    return details, 'success'