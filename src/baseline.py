import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SEED = 42

def make_model(name):
    if name == "svm":
        return LinearSVC(random_state=SEED)
    return LogisticRegression(max_iter=1000, random_state=SEED)

def run_baseline(df, model="lr"):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2,
        stratify=df["label"], random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)   # learn vocabulary from TRAIN only
    Xte = vec.transform(X_test)        # apply it to test
    clf = make_model(model).fit(Xtr, y_train)
    pred = clf.predict(Xte)
    f1 = f1_score(y_test, pred, average="macro")
    print("macro-F1:", round(f1, 4))
    print(classification_report(y_test, pred, target_names=["real", "fake"]))
    print(confusion_matrix(y_test, pred))
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("top REAL features:", names[order[:30]].tolist())
    print("top FAKE features:", names[order[-30:]].tolist())
    return f1
def run_time_split(df, cutoff="2017-07-01", start="2016-01-01", end="2017-12-31", model="lr"):
    d = df[(df["date"] >= start) & (df["date"] <= end)]
    train, test = d[d["date"] < cutoff], d[d["date"] >= cutoff]
    print("train:", train["label"].value_counts().to_dict(),
          "test:", test["label"].value_counts().to_dict())
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    clf = make_model(model).fit(vec.fit_transform(train["text"]), train["label"])
    pred = clf.predict(vec.transform(test["text"]))
    print("macro-F1:", round(f1_score(test["label"], pred, average="macro"), 4))
    print(classification_report(test["label"], pred, target_names=["real", "fake"]))
    print(confusion_matrix(test["label"], pred))

def f1_over_seeds(df, model="lr", seeds=(0, 1, 2, 3, 4)):
    scores = []
    for s in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["text"], df["label"], test_size=0.2,
            stratify=df["label"], random_state=s)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                              max_features=50000, sublinear_tf=True)
        clf = make_model(model).fit(vec.fit_transform(X_tr), y_tr)
        scores.append(f1_score(y_te, clf.predict(vec.transform(X_te)), average="macro"))
    print(round(np.mean(scores), 4), "+/-", round(np.std(scores), 4))
    return scores

def run_cross(isot_df, ext_df, model="lr"):
    X_tr, X_te, y_tr, y_te = train_test_split(
        isot_df["text"], isot_df["label"], test_size=0.2,
        stratify=isot_df["label"], random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    clf = make_model(model).fit(vec.fit_transform(X_tr), y_tr)
    for name, X, y in [("ISOT test (in-domain)", X_te, y_te),
                       ("WELFake external", ext_df["text"], ext_df["label"])]:
        pred = clf.predict(vec.transform(X))
        print("===", name, "===")
        print("macro-F1:", round(f1_score(y, pred, average="macro"), 4))
        print(classification_report(y, pred, target_names=["real", "fake"]))
        print(confusion_matrix(y, pred))

def fit_isot(isot_df, model="lr", seed=SEED):
    X_tr, _, y_tr, _ = train_test_split(
        isot_df["text"], isot_df["label"], test_size=0.2,
        stratify=isot_df["label"], random_state=seed)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    clf = make_model(model).fit(vec.fit_transform(X_tr), y_tr)
    return vec, clf