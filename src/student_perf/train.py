
import argparse
import joblib
import pandas as pd
from .model import train_eval
from .utils import ARTIFACTS_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV data")
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "model.joblib"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    model, r2 = train_eval(df)
    joblib.dump(model, args.out)
    print(f"Saved model to {args.out} (R2={r2:.3f})")

if __name__ == "__main__":
    main()
