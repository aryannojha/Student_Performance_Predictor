
import argparse
import json
import joblib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.joblib")
    ap.add_argument("--json", required=True, help="Path to JSON with features")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    X = pd.DataFrame([data])
    model = joblib.load(args.model)
    pred = float(model.predict(X)[0])
    print(json.dumps({"prediction": pred}, indent=2))

if __name__ == "__main__":
    main()
