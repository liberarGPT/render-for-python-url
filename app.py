from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze():
    return jsonify({'status': 'success', 'symbol': 'AAPL'})

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'status': 'success', 'prediction': 'BUY'})

@app.route('/backtest', methods=['POST'])
def backtest():
    return jsonify({'status': 'success', 'results': {'return': 15.5}})

# Vercel handler
def handler(request):
    return app(request.environ, lambda *args: None)