import pandas
import os
import numpy
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib

def train_model(file):
    # read file, replace nan with mean value
    df = pandas.read_csv(file)

    COLUMN_INFO = {
        'Pregnancies': 'int64',
        'Glucose': 'int64',
        'BloodPressure': 'int64',
        'SkinThickness': 'int64',
        'Insulin': 'int64',
        'BMI': 'float64',
        'DiabetesPedigreeFunction': 'float64',
        'Age': 'int64',
        'Outcome': 'int64'
    }
    required_columns = tuple(COLUMN_INFO.keys())
    # required_columns = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome')
    # print(required_columns)

    current_columns = list(df.columns)
    if len(required_columns) != len(current_columns):
        # print('len')
        return None
    for c in required_columns:
        if c not in current_columns:
            # print(str(c) + ' not found')
            return None
        else:
            if df[c].dtype != COLUMN_INFO[c]:
                # print(f'{c}-{str(df[c].dtype)} and column_info-{COLUMN_INFO[c]}')
                return None
        
    df.fillna(df.mean(), inplace=True)

    # minmax scaler
    mscaler = MinMaxScaler().set_output(transform='pandas')

    # model
    model1 = GaussianNB()
    model2 = LogisticRegression()

    # features and label
    y = df[['Outcome']]
    x = df.drop(columns=['Outcome'])

    # normalize features
    x = mscaler.fit_transform(x)

    # split train test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    y_train = numpy.ravel(y_train)

    # train
    model1 = model1.fit(x_train, y_train)
    model2 = model2.fit(x_train, y_train)

    current = os.path.dirname(__file__)
    current = os.path.join(current, 'saves')

    joblib.dump(model1, os.path.join(current, 'user_trained_model1.joblib'))
    joblib.dump(model2, os.path.join(current, 'user_trained_model2.joblib'))
    joblib.dump(mscaler, os.path.join(current, 'user_trained_mscaler.joblib'))

    pred1 = model1.predict(x_test)
    pred2 = model2.predict(x_test)

    accuracy1 = accuracy_score(pred1, y_test)
    accuracy2 = accuracy_score(pred2, y_test)

    precision1 = precision_score(pred1, y_test)
    precision2 = precision_score(pred2, y_test)

    recall1 = recall_score(pred1, y_test)
    recall2 = recall_score(pred2, y_test)

    confusion_matrix1 = confusion_matrix(pred1, y_test, labels=[0, 1])
    confusion_matrix2 = confusion_matrix(pred2, y_test, labels=[0, 1])

    return {
        'GaussianNB': {
            'Accuracy': accuracy1,
            'Precision': precision1,
            'Recall': recall1,
            'Confusion Matrix': confusion_matrix1,
            'Confusion Matrix Labels': ['No Diabetes', 'Diabetes'],
        },
        'LogisticRegression': {
            'Accuracy': accuracy2,
            'Precision': precision2,
            'Recall': recall2,
            'Confusion Matrix': confusion_matrix2,
            'Confusion Matrix Labels': ['No Diabetes', 'Diabetes'],
        }
    }