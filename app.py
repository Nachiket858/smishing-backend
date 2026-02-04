from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# ==========================
# LOAD SIGNATURE KEYWORDS
# ==========================
with open("signatures.txt", "r") as f:
    signatures = set(line.strip().lower() for line in f if line.strip())

# ==========================
# LOAD ML MODEL & VECTORIZER
# ==========================
model = joblib.load("random_forest_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==========================
# ROOT ROUTE (HTML CHECK)
# ==========================
@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spam Detection API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e5e7eb;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .card {
                background: #020617;
                padding: 30px 40px;
                border-radius: 12px;
                box-shadow: 0 0 30px rgba(0,0,0,0.6);
                text-align: center;
            }
            h1 {
                color: #22c55e;
            }
            p {
                color: #94a3b8;
            }
            code {
                background: #020617;
                padding: 4px 8px;
                border-radius: 6px;
                color: #38bdf8;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🚀 Spam Detection API is Running</h1>
            <p>Server Status: <strong>ACTIVE</strong></p>
            <p>Health Check: <code>/health</code></p>
            <p>Prediction Endpoint: <code>/predict</code></p>
        </div>
    </body>
    </html>
    """

# ==========================
# HEALTH CHECK ROUTE
# ==========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "alive",
        "message": "Server is running fine"
    }), 200

# ==========================
# SPAM PREDICTION ROUTE
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "").lower()

        if not message:
            return jsonify({
                "spam": False,
                "reason": "Empty message received"
            }), 400

        # --------------------------
        # Signature-based detection
        # --------------------------
        matched_keywords = [kw for kw in signatures if kw in message]
        if matched_keywords:
            return jsonify({
                "spam": True,
                "method": "signature",
                "reason": f"Matched keywords: {', '.join(matched_keywords)}"
            }), 200

        # --------------------------
        # ML-based detection
        # --------------------------
        vector = vectorizer.transform([message])
        prediction = model.predict(vector)[0]

        return jsonify({
            "spam": bool(prediction),
            "method": "model",
            "reason": "Predicted by ML model"
        }), 200

    except Exception as e:
        return jsonify({
            "spam": False,
            "error": str(e)
        }), 500

# ==========================
# MAIN ENTRY POINT
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
