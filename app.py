from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import threading
import time
import requests

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
    </head>
    <body style="font-family:Arial;background:#0f172a;color:#e5e7eb;text-align:center;padding-top:100px;">
        <h1>🚀 Spam Detection API is Running</h1>
        <p>Status: ACTIVE</p>
        <p>/health | /predict</p>
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

        # Signature-based detection
        matched_keywords = [kw for kw in signatures if kw in message]
        if matched_keywords:
            return jsonify({
                "spam": True,
                "method": "signature",
                "reason": f"Matched keywords: {', '.join(matched_keywords)}"
            }), 200

        # ML-based detection
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
# SELF-PING BACKGROUND THREAD
# ==========================
def self_ping():
    url = "https://smishing-backend-en4u.onrender.com/health"
    while True:
        try:
            requests.get(url, timeout=10)
            print("✅ Self-ping sent")
        except Exception as e:
            print("❌ Self-ping failed:", e)
        time.sleep(60)  # 1 minute

# ==========================
# MAIN ENTRY POINT
# ==========================
if __name__ == "__main__":
    threading.Thread(target=self_ping, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
