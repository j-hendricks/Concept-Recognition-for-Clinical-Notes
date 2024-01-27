import pandas as pd
import flask
from flask import Flask, render_template, request
import json
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def index():
 return flask.render_template('index.html')

def ValuePredictor(text):
#  print('before:',to_predict_list)
#  to_predict = np.array(to_predict_list).reshape(1,53)
#  print('after:',to_predict)
#  to_predict = to_predict_list.reshape(1,53)
   print("This is the prediction numpy array:",text)
#  loaded_model = pickle.load(open('model.pkl','rb'))
   ner_pipeline = pipeline("ner",model='model_v1.1')
   result = ner_pipeline(text)
   print('result:',result)
   return [f'{res["word"]}: {res["entity"][2:]}' for res in result]

@app.route('/predict',methods = ['POST'])
def result():

 if request.method == 'POST':

    to_predict_list = request.form.to_dict()

    result = ValuePredictor(to_predict_list['notes'])

    
 return render_template('index.html',prediction=result)

if __name__ == "__main__":
 app.run(debug=True)