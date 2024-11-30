from flask import Flask, render_template, request, jsonify
import test
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
    prediction = test.predictImage(image)
    return prediction
    # return "RAHHHH"

if __name__ == '__main__':
    app.run(debug=True)