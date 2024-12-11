from flask import Flask, render_template, request, jsonify
import sys
import os
from test import *
# Add the train_model folder to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_model')))

#from test import predictImage  # Import predictImage from test.py
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/takeImage', methods=['POST'])
def takeImage():
    if 'image' not in request.files:
        return jsonify({"no image"}), 400
    image = request.files['image']
    # hopefully returns a tuple that includes the predicted classification and the percent.
    print(predictImage)
    prediction = predictImage(image)
    print(prediction)
    predict, confidence = prediction[0], prediction[1]
    # predict, confidence = "Card", "20"

    return render_template('result.html', predicted_class=predict, confidence=confidence)

    # return prediction
    # return "RAHHHH"

if __name__ == '__main__':
    app.run(debug=True)