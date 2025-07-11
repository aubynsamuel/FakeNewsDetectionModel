import gc
from flask import Flask, render_template, request, jsonify

from deploy.index import FakeNewsDetector

app = Flask(__name__)

analyzer = FakeNewsDetector()


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
        result = analyzer.comprehensive_verify(headline)

        return result

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
