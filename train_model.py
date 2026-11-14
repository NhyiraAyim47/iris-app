"""Train the iris Decision Tree (same settings as the notebook) and export it
as iris_model.json, which the web app reads. Run:  python train_model.py"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

CSV = Path("iris.csv")
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

df = pd.read_csv(CSV if CSV.exists() else URL)
X, y = df[FEATURES], df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train.values, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test.values))
print(f"Test accuracy: {accuracy:.3f}")

classes = [str(c) for c in model.classes_]
t = model.tree_
export = {
    "features": FEATURES,
    "classes": classes,
    "accuracy": round(float(accuracy), 4),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "tree": {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": [float(v) for v in t.threshold],
        "value": [v[0].tolist() for v in t.value],
    },
    # petal measurements of all 150 flowers, for the scatter plot
    "points": [
        [float(r.petal_length), float(r.petal_width), classes.index(r.species)]
        for r in df.itertuples()
    ],
}
Path("iris_model.json").write_text(json.dumps(export, separators=(",", ":")))
print("Saved iris_model.json")
