import mlflow

mlflow.set_experiment("iris-classification")
mlflow.sklearn.autolog()

from sklearn.datasets import load_iris

data = load_iris()

features, labels = data['data'], data['target']\

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=0)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

with mlflow.start_run() as run:
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)

    mlflow.log_metric("score", score)
