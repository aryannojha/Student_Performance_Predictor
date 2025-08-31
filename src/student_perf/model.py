
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from .features import NUMERICAL, CATEGORICAL

def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categoric = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERICAL),
            ("cat", categoric, CATEGORICAL),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe

def train_eval(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[Pipeline, float]:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    from .features import split_xy

    X, y = split_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipe = build_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    r2 = r2_score(y_val, preds)
    return pipe, r2
