import pandas as pd
import pickle

def data_preprocess(data):
    data['iter'] = 0
    data['prev'] = 0
    data['mean_prev'] = 0
    data['days_from'] = 0
    data['month_from'] = 0
    for i in range(data.shape[0]):
        loc = data[(data['ID (Идентификатор Клиента)'] == data['ID (Идентификатор Клиента)'].iloc[i]) &
                                    (data['Дата заявки'] < data['Дата заявки'].iloc[i])]
        data['iter'].iloc[i] = loc.shape[0]
        data['prev'].iloc[i] = loc['Target (90 mob 12)'].iloc[-1] if loc.shape[0] > 0 else np.nan
        data['mean_prev'].iloc[i] = loc['Target (90 mob 12)'].mean() if loc.shape[0] > 0 else np.nan
        data['days_from'].iloc[i] = -(loc['Дата заявки'].iloc[-1] - data['Дата заявки'].iloc[i]).days if loc.shape[0] > 0 else np.nan
    data['month_from'] = data['days_from'] // 30
    data['month'] = data['Дата заявки'].dt.month
    data['weekday'] = data['Дата заявки'].dt.dayofweek
    data['season'] = data['month'].apply(get_season)
    data['month_concat'] = (data['Дата заявки'].dt.to_period('M') + 1).dt.to_timestamp()
    data = data.set_index(['ID (Идентификатор Клиента)','month_concat'])
    return data


def group_mcc(x):
    return 'Agricultural Services' if x in range(0, 1500) else 'Contracted Services' if x in range(1500, 3000) else \
        'Transportation Services' if x in range(4000, 4800) else 'Utility Services' if x in range(4800, 5000) else\
        'Retail Outlet Services' if x in range(5000, 5600) else 'Clothing Stores' if x in range(5600, 5700) else \
        'Miscellaneous Stores' if x in range(5700, 7300) else 'Business Services' if x in range(7300, 8000) else \
        'Professional Services and Membership Organizations' if x in range(8000, 9000) else \
        'Airlines' if x in range(3000,3300) else 'Car Rental' if x in range(3300, 3500) else \
        'Lodging' if x in range(3500, 4000) else np.nan


def data_all_preprocess(data_all, time, ids):
    data_all = data_all.rename(columns={'Идентификатор Клиента':'ID (Идентификатор Клиента)'})
    data_all['Сумма транзакции'] = data_all['Сумма транзакции'].astype('float32')
    data_all['group_mcc'] = data_all['MCC-код транзакции'].apply(group_mcc)
    data_all['month_concat'] = pd.to_datetime(data_all['Дата транзакции'].dt.year.astype('str') + '-' + data_all['Дата транзакции'].dt.month.astype('str') + '-01')

    data_all[data_all['month_concat'] >= time]
    data_all[data_all['ID (Идентификатор Клиента)'].isin(ids)]

    data_all = data_all[data_all['Валюта транзакции'] == 'Russian Ruble']
    data_all_deb = data_all[data_all['Тип карты'] == 'Дебетовая карта']
    data_all_cre = data_all[data_all['Тип карты'] != 'Дебетовая карта']
    return data_all_deb, data_all_cre


def col_rename(data, suf, prefix):
    new_cols = []
    for col in data.columns:
        loc_col = [prefix, suf]
        if isinstance(col, tuple):
            for key in col:
                loc_col.append(key)
        else:
            loc_col.append(col)
        new_cols.append(tuple(loc_col))
    return new_cols


def gen_rolling(data, roll):
    index = data.index
    data = data.reset_index().groupby('ID (Идентификатор Клиента)')[data.columns[2:]].rolling(roll).agg(['min','max','mean','sum'])
    data.index = index
    return data


def join_datas(data, data_all, prefix):
    group_cols = ['group_mcc', 'Признак принадлежности транзакции в банкомате АББ', 'Признак принадлежности транзакции в стороннем банкомате', 'Бизнес-тип операции', 'Валюта платежной системы', 'Направление транзакции']
    func = ['count','mean','median','sum']

    to_join = data_all.groupby(['ID (Идентификатор Клиента)','month_concat'])['Сумма транзакции'].agg(func)
    to_join.columns = col_rename(to_join, 'prev', prefix)
    data = data.join(to_join)

    for col in group_cols:
        to_join = data_all.groupby(['ID (Идентификатор Клиента)','month_concat',col])['Сумма транзакции'].agg(['count','sum']).unstack()
        to_join.columns = col_rename(to_join, col, prefix)
        to_join = gen_rolling(to_join, 3)
        data = data.join(to_join)

    return data


def pca_decompose(data):
    from sklearn.decomposition import PCA

    pca = PCA(25)
    data = pca.fit_transform(data.drop('Target (90 mob 12)', axis=1).fillna(0))
    pickle.dump(pca, open('pca.pickle', 'wb'))
    return data


def get_season(x):
    return 1 if x in [12,1,2] else 2 if x in [3,4,5] else 3 if x in [6,7,8] else 4


def pred_transform(data):
    data = pd.Series(data)
    data[data > 0.025] = 0.95
    return data


def train_model(x, y):
    from lightgbm import Dataset, train

    params = {'learning_rate': 0.09,
              'num_leaves': 63,
              'objective': 'binary',
              'min_child_samples': 2,
              'n_jobs': -1,
              'is_unbalance': True
              }
    train_set = Dataset(x, label=y)
    clf = train(params, train_set, num_boost_round=50)

    pickle.dump(clf, open('clf.pickle', 'wb'))


def predict_model(data):
    clf = pickle.load(open('clf.pickle', 'rb'))
    pca = pickle.load(open('pca.pickle', 'rb'))

    x = pca.transform(data)
    pred = clf.predict(x)
    pred = pred_transform(pred)
    return pred


def preprocess(data, data_all):
    data_all = data_all.iloc[1:]
    data_all = data_all.sort_values(by='Дата транзакции').reset_index(drop=True)

    data = data_preprocess(data)
    data_all, credit = data_all_preprocess(data_all,
                                           str(data.index.get_level_values(1).to_period('M').min() - 6) + '-01',
                                           data.index.get_level_values(0).unique())

    data = join_datas(data, data_all, 'debit')
    data = join_datas(data, credit, 'credit')
    data = data.reset_index()
    return data