from  flask import Flask,render_template,url_for,redirect,request
import pickle
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np


with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

with open('encode_sentiment.pkl','rb') as file:
    encoder=pickle.load(file)

app=Flask(__name__)


@app.route("/" , methods=['GET','POST'])
def index():

    if request.method=='POST':
        text=request.form['textInput']
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        model = load_model("sentiment.h5")
        prediction=model.predict(padded)

        pred_label = encoder.inverse_transform(prediction)

        return render_template('base.html',output=pred_label[0][0])

    else:
        return render_template("base.html")




if __name__ in "__main__":
    app.run(debug=True)
