import pytest
from src.classify import ClassificationPipeline
import pandas as pd


@pytest.fixture
def clf(mocker):
    """Returns a ClassificationPipeline object with a mocked Model object"""
    mock_model = mocker.patch('src.classify._Model', autospec=True)
    mock_model_instance = mock_model.return_value

    def mocked_predict(texts):
        if isinstance(texts, str):
            return ['test']
        else:
            return ['test'] * len(texts)

    mock_model_instance.predict.side_effect = mocked_predict
    return ClassificationPipeline('cpu')


def test_predict_single_string(clf):
    # Arrange
    text = "US economy grows by 6.4% in first quarter"
    expected = pd.DataFrame({
        'headline': [text],
        'model topic': ['test'],
        'sentiment': ['test']
    })
    # Act
    result = clf.predict(text)
    # Assert
    assert result.equals(expected)


def test_predict_list_of_strings(clf):
    # Arrange
    texts = [
        "US economy grows by 6.4% in first quarter",
        "initial jobless claims lowest since March 2020"
    ]
    expected = pd.DataFrame({
        'headline': texts,
        'model topic': ['test', 'test'],
        'sentiment': ['test', 'test']
    })
    # Act
    result = clf.predict(texts)
    # Assert
    assert result.equals(expected)
