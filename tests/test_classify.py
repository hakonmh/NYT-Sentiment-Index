import pytest
from src.classify import ClassificationPipeline, classify_latest, classify_full_history
import pandas as pd
from .fixtures import _fs_read_file, RAW_DATA_PATH, CLASSIFIED_DATA_PATH

FILES = ['2022-01.csv', '2022-02.csv', '2022-03.csv']


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
    return ClassificationPipeline(device='cpu')


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


def test_predict_dataframe(clf, fs):
    # Arrange
    texts = [
        "US economy grows by 6.4% in first quarter",
        "initial jobless claims lowest since March 2020"
    ]
    raw_data = pd.DataFrame({'headline': texts, 'topic': ['topic', 'topic']})
    raw_data.index = pd.date_range('2022-01-01', periods=2, freq='D')
    expected = pd.DataFrame({
        'headline': texts,
        'topic': ['topic', 'topic'],
        'model topic': ['test', 'test'],
        'sentiment': ['test', 'test']
    })
    expected.index = pd.DatetimeIndex(['2022-01-01', '2022-01-02'])
    # Act
    result = clf.predict(raw_data)
    # Assert
    assert result.equals(expected)


def test_classify_full_history_overwrite(clf, fs):
    """Tests if classify_full_history correctly overwrites all files in the
    classified-data dir when overwrite=True
    """
    # Arrange
    _setup_fs(fs)
    expected_file_content = _create_classified_file_content(num_rows=3)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-01.csv', contents=expected_file_content)
    partially_filled_file_content = _create_classified_file_content(num_rows=2)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-02.csv', contents=partially_filled_file_content)
    # Act
    classify_full_history(RAW_DATA_PATH, CLASSIFIED_DATA_PATH, overwrite=True)
    # Assert
    assert fs.listdir(CLASSIFIED_DATA_PATH) == FILES
    for f in FILES:
        file_path = f'{CLASSIFIED_DATA_PATH}/{f}'
        assert _fs_read_file(file_path) == expected_file_content


def test_classify_full_history_not_overwrite(clf, fs):
    """Tests if classify_full_history correctly overwrites only unclassified files in the
    classified-data dir when overwrite=False
    """
    # Arrange
    _setup_fs(fs)
    expected_file_content = _create_classified_file_content(num_rows=3)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-01.csv', contents=expected_file_content)
    partially_filled_file_content = _create_classified_file_content(num_rows=2)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-02.csv', contents=partially_filled_file_content)
    # Act
    classify_full_history(RAW_DATA_PATH, CLASSIFIED_DATA_PATH, overwrite=False)
    # Assert
    assert fs.listdir(CLASSIFIED_DATA_PATH) == FILES
    assert _fs_read_file(f'{CLASSIFIED_DATA_PATH}/2022-01.csv') == expected_file_content
    assert _fs_read_file(f'{CLASSIFIED_DATA_PATH}/2022-02.csv') == partially_filled_file_content
    assert _fs_read_file(f'{CLASSIFIED_DATA_PATH}/2022-03.csv') == expected_file_content


def test_classify_latest(clf, fs):
    """Tests if classify_latest correctly classifies unclassified and partially classified
    files in the raw-data directory"""
    # Arrange
    _setup_fs(fs)
    # write fully classified content to first file in classified-data dir
    expected_file_content = _create_classified_file_content(num_rows=3)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-01.csv', contents=expected_file_content)
    # write partially classified content to second file in classified-data dir
    partially_filled_file_content = _create_classified_file_content(num_rows=2)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-02.csv', contents=partially_filled_file_content)

    # Act
    classify_latest(RAW_DATA_PATH, CLASSIFIED_DATA_PATH)
    # Assert
    assert fs.listdir(CLASSIFIED_DATA_PATH) == FILES
    for f in FILES:
        file_path = f'{CLASSIFIED_DATA_PATH}/{f}'
        assert _fs_read_file(file_path) == expected_file_content


def _setup_fs(fs):
    """Setup fake filesystem with 3 files of unclassified data in raw-data dir"""
    fs.create_dir(RAW_DATA_PATH)
    fs.create_dir(CLASSIFIED_DATA_PATH)

    # write raw_file_content to 3 files in raw-data dir
    raw_file_content = _create_raw_file_content(num_rows=3)
    for f in FILES:
        fs.create_file(f'{RAW_DATA_PATH}/{f}', contents=raw_file_content)


def _create_raw_file_content(num_rows=1):
    """Creates a string with the content for an unclassified data file"""
    file_content = "date,headline,topic\n"
    for i in range(num_rows):
        date = pd.Timestamp(f'2022-01-01') + pd.Timedelta(days=i)
        date = date.strftime('%Y-%m-%d 00:00:00')
        line = f"{date},headline,none\n"
        file_content += line
    return file_content


def _create_classified_file_content(num_rows=1):
    """Creates a string with the content for a classified data file"""
    file_content = "date,headline,topic,model topic,sentiment\n"
    for i in range(num_rows):
        date = pd.Timestamp(f'2022-01-01') + pd.Timedelta(days=i)
        date = date.strftime('%Y-%m-%d 00:00:00')
        line = f"{date},headline,none,test,test\n"
        file_content += line
    return file_content
