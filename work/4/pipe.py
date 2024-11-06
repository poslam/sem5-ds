import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df["hour"] = df[self.datetime_column].dt.hour
        df["day"] = df[self.datetime_column].dt.day
        df["month"] = df[self.datetime_column].dt.month
        df["year"] = df[self.datetime_column].dt.year
        df["dayofweek"] = df[self.datetime_column].dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df.drop(
            columns=[self.datetime_column],
            inplace=True,
        )

        return df


class WeightTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_weights):
        self.column_weights = column_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, weight in self.column_weights.items():
            if col in X.columns:
                X[col] = X[col] * weight
        return X


data = pd.read_csv("C:/demiskira/projects/fefu/sem5/ds/work/4/bike_sharing_demand.csv")
data["datetime"] = pd.to_datetime(data["datetime"])

X = data.drop(columns=["count", "casual", "registered"], axis=1)
y = data["count"]

numeric_cols = [
    "datetime",
    "atemp",
    "humidity",
    "windspeed",
]
categorical_cols = ["season", "weather", "holiday", "workingday"]

numeric_transformer = Pipeline(
    steps=[
        ("datetime_features", DatetimeTransformer(datetime_column="datetime")),
        (
            "weight_features",
            WeightTransformer(column_weights={"atemp": 3}),
        ),
        ("imputer", SimpleImputer(strategy="mean")),
        ("polynomial_featurer", PolynomialFeatures(4)),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LinearRegression(n_jobs=4),
        ),
    ]
)

(X_train, X_test, y_train, y_test) = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=69,
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("r2:\t", r2)
print("rmse:\t", rmse)
