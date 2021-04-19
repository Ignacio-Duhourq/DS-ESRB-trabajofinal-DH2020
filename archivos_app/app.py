import numpy as np
import pandas as pd
import json
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelo_video.pkl', 'rb'))
showPredictions = None
df = pd.read_csv('video_games_test_data.csv')

def get_features(df, game_name):
    row = df[df.title == game_name].iloc[-1]
    return pd.DataFrame(row).T

@app.route('/')

def home():
    return render_template("index.html", showPredictions= None)

@app.route('/predict', methods=['GET'])

def predict():
    game_name =request.args['game_name']
    game_to_predict = get_features(df, game_name)
    game_to_predict = game_to_predict.drop(columns = ['title','esrb_rating'])
    game_to_predict = game_to_predict.astype('int32')

    prediction = model.predict(game_to_predict)[0]

    a = list(zip(model.classes_, model.predict_proba(game_to_predict)[0]))

    return render_template('index.html', showPredictions= True, prediction = prediction, game_name = game_name, e_proba = "{:.3%}".format(float(a[0][1])), et_proba = "{:.3%}".format(float(a[1][1])) , m_proba = "{:.3%}".format(float(a[2][1])), t_proba =  "{:.3%}".format(float(a[3][1])))

if __name__ == '__main__':
    app.run(debug=True)