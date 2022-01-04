from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X, y, **kwargs):
    clf = RandomForestClassifier(random_state=42, **kwargs)
    clf.fit(X, y)
    return clf


def predict_random_forest(X, model, **kwargs):
    return model.predict(X)
