import pickle
import pandas as pd
from flask import Flask, request, Response
from olist.Olist import Olist

app = Flask(__name__)

# load model
model = pickle.load(open('C:/Users/Pedro/Desktop/Codar/Jupyter/Projetos/Olist/random_forest_finalized.pkl', 'rb'))

@app.route('/olist/predict', methods=['POST'])

def olist_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):  # Unique example
            test_raw = pd.DataFrame(test_json, index=[0])

        else:  # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instantiate Olist class
        pipeline = Olist()

        # data cleaning
        df = pipeline.data_cleaning(test_raw)

        # feature enginering
        df = pipeline.feature_enginering(df)

        # data preparation
        df = pipeline.data_prep(df)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df)

        return df_response

    else:
        return Reponse('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('192.168.0.26')
