from flask import Flask, request
import pandas as pd
from lib import preprocess, predict_model, pca_decompose, train_model, upload

app = Flask(__name__)
if __name__ == '__main__':
    app.run()

@app.route('/upload/data', methods=['POST'])
def upload_data():
    if request.method == 'POST':
        file = request.files['data.xlsx']
        new_data = pd.read_excel(file)
        upload(new_data, 'data')
    return 'Loaded'


@app.route('/upload/trans', methods=['POST'])
def upload_trans():
    if request.method == 'POST':
        file = request.files['trans.xlsx']
        new_data = pd.read_excel(file)
        upload(new_data, 'trans')
    return 'Loaded'


@app.route('/predict', methods=['POST'])
def predict():
    import datetime
    from json import dumps

    data = {}
    if request.method == 'POST':
        data['ID (Идентификатор Клиента)'] = request.form['ID (Идентификатор Клиента)']
        data['Дата транзакции'] = datetime.datetime.now()

        data = pd.DataFrame(data)
        data_all = pd.read_excel('trans.xlsx')

        data = preprocess(data, data_all)
        pred = predict_model(data)
        return dumps(pred.tolist())


@app.route('/fit')
def fit():
    data = pd.read_excel('data.xlsx')
    data_all = pd.read_excel('trans.xlsx')

    data = data.sort_values(by=['Дата заявки']).reset_index(drop=True)
    data = preprocess(data, data_all)

    data = data.drop(['ID (Идентификатор Заявки)', 'Дата заявки', 'month_concat', 'ID (Идентификатор Клиента)'], axis=1)
    y = data['Target (90 mob 12)']
    x = pca_decompose(data.drop('Target (90 mob 12)', axis=1))

    train_model(x, y)
    return 'Fitted'