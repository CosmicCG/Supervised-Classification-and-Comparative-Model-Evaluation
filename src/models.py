"""
Shared model definitions + preprocessing pipeline.

Every model gets the SAME preprocessing so the comparison is fair:
  numeric/ordinal/binary -> median impute -> standard scale
  categorical            -> most-frequent impute -> one-hot

Imbalance strategies:
  "none"          - plain model
  "class_weight"  - class_weight="balanced" (LogReg, Tree, RF only;
                    KNN and Naive Bayes have no class_weight option)
  "smote"         - SMOTE oversampling inside the pipeline (needs imbalanced-learn).
                    It is applied only to the training folds, never the test set.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    HAS_IMBLEARN = False

SEED = 42
MODEL_NAMES = ["LogReg", "DecisionTree", "KNN", "NaiveBayes", "RandomForest"]
SUPPORTS_CLASS_WEIGHT = {"LogReg", "DecisionTree", "RandomForest"}


def make_model(name: str, class_weight=None):
    if name == "LogReg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=SEED)
    if name == "DecisionTree":
        # max_depth limited on purpose; see validation curve for the bias-variance story
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                      class_weight=class_weight, random_state=SEED)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=25, weights="uniform", n_jobs=-1)
    if name == "NaiveBayes":
        return GaussianNB()
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                      class_weight=class_weight, n_jobs=-1, random_state=SEED)
    raise ValueError(name)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers = [("num", num, numeric)]
    if categorical:
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers)


def make_pipeline(name: str, numeric: list[str], categorical: list[str], strategy: str = "none"):
    """Returns an unfitted pipeline, or None if the combination doesn't apply."""
    pre = make_preprocessor(numeric, categorical)
    if strategy == "class_weight":
        if name not in SUPPORTS_CLASS_WEIGHT:
            return None
        return Pipeline([("pre", pre), ("model", make_model(name, "balanced"))])
    if strategy == "smote":
        if not HAS_IMBLEARN:
            return None
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=SEED)),
                            ("model", make_model(name))])
    return Pipeline([("pre", pre), ("model", make_model(name))])
