import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)

model=pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('forest_fire.html')


@app.route('/predict1', methods=['POST','GET'])
def predict1():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict(final)
    output = round(prediction[0], 2)
    return render_template('forest_fire.html', prediction_='The house price is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)