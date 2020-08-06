import pandas as pd
import pickle

def get_season(x):
    return 1 if x in [12,1,2] else 2 if x in [3,4,5] else 3 if x in [6,7,8] else 4


def date_features(data):
    data['month'] = data['RequestDate'].dt.month
    data['season'] = data['month'].apply(get_season)
    data['is_month_start'] = data['RequestDate'].dt.is_month_start
    data['is_month_end'] = data['RequestDate'].dt.is_month_end
    data['dayofweek'] = data['RequestDate'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'] >= 5
    data = data.drop('RequestDate', axis=1)
    return data


def get_obj_cols(data):
    return data.dtypes[data.dtypes == 'O'].index


def pred_transform(data):
    data = pd.Series(data)
    data[data > 0.1] = 0.95
    return data


def train_model(data):
    from lightgbm import Dataset, train

    params = {'learning_rate': 0.02,
              'num_leaves': 15,
              'objective': 'binary',
              'min_child_samples': 10,
              'n_jobs': -1,
              'is_unbalance': False
              }
    train_set = Dataset(data.drop('Target', axis=1), label=data['Target'])
    clf = train(params, train_set, num_boost_round=300)

    pickle.dump(clf, open('clf.pickle', 'wb'))


def predict_model(data):
    clf = pickle.load(open('clf.pickle', 'rb'))
    pred = clf.predict(data)
    pred = pred_transform(pred)
    return pred


def preprocess(data):
    obj_cols = get_obj_cols(data)
    float_cols = list(set(data.columns[2:]) - set(obj_cols))
    data[float_cols] = data[float_cols].astype('float32')

    data = date_features(data)

    data = data.drop(obj_cols[0], axis=1)
    data[obj_cols[1:]] = data[obj_cols[1:]].astype('category')
    return data