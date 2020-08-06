import pandas as pd
import numpy as np
import pickle


def get_season(x):
  return 1 if x in [12,1,2] else 2 if x in [3,4,5] else 3 if x in [6,7,8] else 4


def preprocess_datetime(df, time_columns):
    for t in time_columns:
        df[t+'_year'] = df[t].dt.year
        df[t+'_month'] = df[t].dt.month
        df[t+'_season'] = df[t+'_month'].apply(get_season)
        df[t+'_is_month_start'] = df[t].dt.is_month_start
        df[t+'_is_month_end'] = df[t].dt.is_month_end
        df[t+'_day'] = df[t].dt.day
        # df[t+'_hour'] = df[t].dt.hour
        df[t+'_dayofweek'] = df[t].dt.dayofweek
        df[t+'_is_weekend'] = df[t+'_dayofweek'] >= 5
        df[t+'_quarter'] = df[t].dt.quarter


def get_regions(df, reg, city_to_regions):
     base = {'Кукморский': 'Кукмор',
          'Мамадышский': 'Мамадыш',
          'Нурлатский': 'Нурлат',
          'Азнакаевский': 'Азнакаево', 
          'Апастовский':'Апастово',
          'Бугульминский': 'Бигульма',
          'Елабужский':'Елабуга',
          'Бавлинский':'Бавлы',
          'Нижегородский':'Нижний Новгород',
          'Чистопольский':'Чистополь',
          'Вятские поляны': 'Вятские Поляны',
          'Новгород Великий':'Новгород'}
     to_cut = {k:k[:-2]  for k in reg if 'ский' in k}

     temp = df.LivingRegionName.copy().astype('object')
     temp[temp.isin(base.keys())] = temp[temp.isin(base.keys())].map(base)
     temp[temp.isin(to_cut.keys())]  = temp[temp.isin(to_cut.keys())] .map(to_cut)
     temp = temp.map(city_to_regions)
     # temp[df.LivingRegionName=='Москва'] = 'Москва'
     assert temp.isna().sum() == 0

     df.LivingRegionName = temp.astype('category')
     df.LivingRegionName = df.LivingRegionName.astype('object').str.replace('Башкирия', '')\
                                    .str.replace('Горьковская', 'обл.')\
                                    .str.replace('(', '').str.replace(')', '')\
                                    .str.replace('Чувашия', 'Чувашская Республика')\
                                    .str.replace('Удмуртия', 'Удмуртская Республика')\
                                    .str.replace('Татарстан', 'Республика Татарстан')\
                                    .str.replace('Марий Эл', 'Республика Марий Эл') \
                                    .str.replace('Башкортостан', 'Республика Башкортостан')\
                                    .str.replace('Санкт-Петербург и область', 'г.Санкт-Петербург')\
                                    .str.replace('Москва и Московская обл.', 'г.Москва')


def get_income_region(x):
    incom_region = pd.read_pickle('income_region.pkl')
    year, period = x.RequestDate_year, x.RequestDate_quarter
    assert np.isin(x.LivingRegionName, incom_region.region)
    return incom_region[(incom_region.region == x.LivingRegionName)][(year,period)].iat[0]


def preprocess_categories(df):
    df.Employment.cat.reorder_categories(['нет данных', 'Сотрудник \ Рабочий \ Ассистент',
                                     'Эксперт\Старший или Ведущий Специалист',
                                     'Главный Специалист\Руководитель среднего звена',
                                     'Руководитель высшего звена', 'Индивидуальный предприниматель'],
                                     ordered=True,inplace=True)
    df.EducationStatus.cat.reorder_categories(['Незаконченное среднее образование','Среднее образование', 
                                           'Среднее специальное образование','Незаконченное высшее образование',
                                          'Высшее образование','Несколько высших образований' ,
                                           'Академическая степень (кандидат наук, доктор наук и т.д.)'],
                                           ordered=True,inplace=True)
    df.kolichestvo_rabotnikov_v_organizacii.cat.reorder_categories(['до 20',  '21-100','101-500', 'более 500'],
                                                                ordered=True,inplace=True)
    df.kolichestvo_detej_mladshe_18.cat.as_ordered(inplace=True)
    df.IncomeDocumentKind.cat.reorder_categories(['2-НДФЛ', 'Выписка по счету', 'Справка по форме Банка от работодателя',
                                               'Иное', 'Нет (не предоставлен)','Нет запрашивали'],
                                              ordered=True, inplace=True)

def preprocess_ini(df):
    df.rename(columns={'ConfirmedMonthlyIncome (Target)': 'target'}, inplace=True)
    df.drop('NaturalPersonID', axis=1, inplace=True)
    
    preprocess_datetime(df, ['RequestDate'])
    df = df.drop('RequestDate', axis=1)
    features_time = [f for f in df.columns if 'RequestDate_' in f]
    
    df.drop(['SignIP.1', 'Employment.1','TypeOfWork.1'], axis=1, inplace=True)
    df.kolichestvo_detej_mladshe_18 = df.kolichestvo_detej_mladshe_18.astype('object')
    features_cat =  df.select_dtypes('object').columns.tolist()

    regions = pd.read_json('russia')
    city_to_regions = regions.set_index('city').to_dict()['region']
    reg = [f for f in df.LivingRegionName.unique() if 'ский' in f]
    city = [f for f in df.LivingRegionName.unique() if f not in reg]
    df['in_region'] = np.where(df.LivingRegionName.isin(reg), 1,0)
    get_regions(df, reg, city_to_regions)

    ###!!!!!!!!!!
    #incom_region = pd.read_excel
    #incom_region.rename(columns={'Unnamed: 0': 'region'}, inplace=True)
    #incom_region.columns = pd.MultiIndex.from_product([np.arange(2013,2021), (1,2,3,4,'total')]).insert(0, incom_region.columns[0])
    #incom_region.region = incom_region.region.str.strip().str.replace('область', 'обл.')
    df['income_region'] = df.apply(get_income_region, axis=1)
    
####!!!!!!!!!!!!
    df = df[(df.otrasl_rabotodatelya.notna())&(df.kolichestvo_rabotnikov_v_organizacii.notna())]
    df.SignIP = df.SignIP.astype('object').fillna(0)
    df[df.SignIP=='ИП', 'TypeOfWork'] = 'Индивидуальный предприниматель'
    df.loc[df.SignIP=='ИП', 'Employment'] = 'Индивидуальный предприниматель'
    df.Employment.fillna('нет данных', inplace=True)
    # сотрудники банка работают на полную ставку - заполнение пропска по типу работы
    df.loc[(df.IsBankWorker=='да')&(df.harakteristika_tekutschego_trudoustrojstva.isna()),
       'harakteristika_tekutschego_trudoustrojstva'] = 'Постоянная, полная занятость'
    df.IncomeDocumentKind = df.IncomeDocumentKind.astype('object').fillna('Нет запрашивали')
    df.Residence = df.Residence.astype('object').fillna('na')
    df.TypeOfWork = df.TypeOfWork.astype('object').fillna('na')
    df['OrgNtoAge'] = (df.OrgStanding_N)/(df.age-18)

    target = df.target
    df[features_cat] = df[features_cat].astype('category')
    df[features_time] = df[features_time].astype('category') 
    preprocess_categories(df)
    to_drop = [ 'RequestDate_year']#+ ['target']
    df.drop(to_drop, axis=1,inplace=True)
   
#######!!!!!! cat for test

    return df

def preprocess(df):
    df = preprocess_ini(df)
    to_drop = ['RequestDate_day']
    df.drop(to_drop, axis=1, inplace=True)
    features_cat = df.select_dtypes('category').columns
    
    # df.CreditSum.fillna(0, inplace=True) #doesn't help
    
    #!!!!!!!!!X,X_test = feature_by_mean('age', ['otrasl_rabotodatelya', 'Employment'], X, X_test)
    #X, X_test = feature_by_mean('OrgStanding_N', ['otrasl_rabotodatelya', 'Employment'], X, X_test)
    #y.reset_index(drop=True, inplace=True)
    
    # X, y = del_by_values(['Иное', 'Нет (не предоставлен)'], 'IncomeDocumentKind', X, y ) #!!!!!
    # X, y = del_by_values([0], 'CreditSum', X,y) # !!!!!

    for f in features_cat:
        df[f] = df[f].astype('category')

    return df


def train_model(data):
    from lightgbm import Dataset, train

    params = {'learning_rate': 0.02,
              'num_leaves': 17,
              'objective': 'mean_squared_error',
              'min_child_samples': 15,
              'n_jobs': -1,
              }
    train_set = Dataset(data.drop('target', axis=1), label=data['target'])
    clf = train(params, train_set, num_boost_round=500)

    pickle.dump(clf, open('clf.pickle', 'wb'))


def predict_model(data):
    clf = pickle.load(open('clf.pickle', 'rb'))
    pred = clf.predict(data)
    return pred

