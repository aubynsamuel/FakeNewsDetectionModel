from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Add this

# from deploy.index import FakeNewsDetector # uncomment if running locally

from gradio_client import Client  # comment out if running locally

app = Flask(__name__)
CORS(app)

# analyzer = FakeNewsDetector()   # uncomment if running locally


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        headline = data.get("headline", "").strip()

        if not headline:
            return jsonify({"error": "Please enter a headline to analyze"})

        if len(headline) < 10:
            return jsonify(
                {"error": "Headline too short. Please enter a more complete headline."}
            )

        # Analyze the headline

        # uncomment if running locally
        # result = analyzer.comprehensive_verify(headline)

        # comment out if running locally
        client = Client("aubynsamuel05/nli_checks")
        result = client.predict(raw_headline=headline, api_name="/predict")

        return result

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
