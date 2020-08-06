from flask import Flask, request
import pandas as pd
from lib import preprocess, train_model, predict_model

app = Flask(__name__)
if __name__ == '__main__':
    app.run()

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['data.xlsx']
        new_data = pd.read_excel(file)
        data = pd.read_excel('data.xlsx')
        data = data.append(new_data, ignore_index=True)
        data.to_excel('data.xlsx')
    return 'Loaded'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        from json import dumps
        data = request.files['data.xlsx']
        data = preprocess(data)
        pred = predict_model(data)
        return dumps(pred.tolist())


@app.route('/fit')
def fit():
    data = pd.read_excel('data.xlsx')
    data = preprocess(data)

    train_model(data)
    return 'Fitted'
