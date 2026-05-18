from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

model = mlflow.pyfunc.load_model("models:/HousingModel/Production")

@app.route("/predict", methods=["POST"])
def predict():
    data = pd.DataFrame(request.json)
    preds = model.predict(data)
    return jsonify(preds.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
