from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, Y_train):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    return model