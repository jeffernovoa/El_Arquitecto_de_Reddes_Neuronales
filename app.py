from flask import Flask, request, render_template, jsonify
from train_mnist import train_and_evaluate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    architecture = data.get('architecture')
    acc = train_and_evaluate(architecture)
    return jsonify({"accuracy": acc, "image_url": "/static/results.png"})

if __name__ == '__main__':
    app.run(debug=True)
