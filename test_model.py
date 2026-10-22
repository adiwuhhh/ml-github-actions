from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from train import train_model


def test_model_trains_successfully():
    model, X_test, y_test, accuracy = train_model()
    assert model is not None
    assert accuracy >= 0.80


def test_predictions_have_correct_length():
    model, X_test, y_test, _ = train_model()
    assert len(model.predict(X_test)) == len(y_test)


def test_predictions_are_valid_classes():
    model, X_test, _, _ = train_model()
    valid_classes = set(load_iris().target)
    assert set(model.predict(X_test)).issubset(valid_classes)


def test_accuracy_is_valid():
    model, X_test, y_test, _ = train_model()
    actual_accuracy = accuracy_score(y_test, model.predict(X_test))
    assert 0.0 <= actual_accuracy <= 1.0
    assert actual_accuracy >= 0.80
