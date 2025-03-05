import pandas

def file_validation(file):
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

    current_columns = list(df.columns)
    flag = True
    err = ''

    if len(required_columns) != len(current_columns):
        flag = False
        err += '\nError: column count mismatch'
        return flag, err
    
    for c in required_columns:
        if c not in current_columns:
            # print(str(c) + ' not found')
            flag = False
            err += f'\nError: missing column - {c}'
        else:
            if df[c].dtype != COLUMN_INFO[c]:
                # print(f'{c}-{str(df[c].dtype)} and column_info-{COLUMN_INFO[c]}')
                flag = False
                err += f'\nError: mismatch datatype - {c} column with {str(df[c].dtype)}, it should be {COLUMN_INFO[c]}'

    return flag, err            