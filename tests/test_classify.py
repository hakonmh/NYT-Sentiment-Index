from .fixtures import _fs_read_file, RAW_DATA_PATH, CLASSIFIED_DATA_PATH

import pandas as pd
from src.classify import ClassificationPipeline, classify_latest, classify_full_history

FILES = ['2022-01.csv', '2022-02.csv', '2022-03.csv']


def mock_model_class_and_cuda(mocker):
    """Mocks the _Model class, its predict method and the torch.cuda.is_available function"""
    def mocked_predict(_, texts):
        if isinstance(texts, str):
            return ['test']
        else:
            return ['test'] * len(texts)

    mock_model_predict = mocker.patch('src.classify._Model.predict', autospec=True)
    mock_model_predict.side_effect = mocked_predict
    mock_model_init = mocker.patch('src.classify._Model.__init__', autospec=True)
    mock_model_init.return_value = None
    mock_cuda_is_available = mocker.patch('torch.cuda.is_available', autospec=True)
    mock_cuda_is_available.return_value = False


def test_predict_single_string(mocker):
    # Arrange
    mock_model_class_and_cuda(mocker)

    text = "US economy grows by 6.4% in first quarter"
    expected = pd.DataFrame({
        'headline': [text],
        'model topic': ['test'],
        'sentiment': ['test']
    })
    # Act
    result = ClassificationPipeline().predict(text)
    # Assert
    assert result.equals(expected)


def test_predict_list_of_strings(mocker):
    # Arrange
    mock_model_class_and_cuda(mocker)

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
    result = ClassificationPipeline().predict(texts)
    # Assert
    assert result.equals(expected)


def test_predict_dataframe(mocker, fs):
    # Arrange
    mock_model_class_and_cuda(mocker)

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
    result = ClassificationPipeline().predict(raw_data)
    # Assert
    assert result.equals(expected)


def test_classify_full_history_overwrite(mocker, fs):
    """Tests if classify_full_history correctly overwrites all files in the
    classified-data dir when overwrite=True
    """
    # Arrange
    mock_model_class_and_cuda(mocker)

    _create_and_write_raw_data(fs)
    file_contents = _create_and_write_classified_data(fs)
    classified_file_content, _ = file_contents

    expected_file_contents = [classified_file_content,
                              classified_file_content,
                              classified_file_content]
    # Act
    classify_full_history(RAW_DATA_PATH, CLASSIFIED_DATA_PATH, overwrite=True)
    # Assert
    assert fs.listdir(CLASSIFIED_DATA_PATH) == FILES
    _assert_file_contents_equal(CLASSIFIED_DATA_PATH, FILES, expected_file_contents)


def test_classify_full_history_not_overwrite(mocker, fs):
    """Tests if classify_full_history correctly overwrites only unclassified
    files in the classified-data dir when overwrite=False
    """
    # Arrange
    mock_model_class_and_cuda(mocker)

    _create_and_write_raw_data(fs)
    file_contents = _create_and_write_classified_data(fs)
    classified_file_content, partially_classified_file_content = file_contents

    expected_file_contents = [classified_file_content,
                              partially_classified_file_content,
                              classified_file_content]
    # Act
    classify_full_history(RAW_DATA_PATH, CLASSIFIED_DATA_PATH, overwrite=False)
    # Assert
    assert fs.listdir(CLASSIFIED_DATA_PATH) == FILES
    _assert_file_contents_equal(CLASSIFIED_DATA_PATH, FILES, expected_file_contents)


def test_classify_latest(mocker, fs):
    """Tests if classify_latest correctly classifies unclassified and partially
    classified files in the raw-data directory
    """
    # Arrange
    mock_model_class_and_cuda(mocker)

    _create_and_write_raw_data(fs)
    file_contents = _create_and_write_classified_data(fs)
    classified_file_content, _ = file_contents

    expected_file_contents = [classified_file_content,
                              classified_file_content,
                              classified_file_content]
    # Act
    classify_latest(RAW_DATA_PATH, CLASSIFIED_DATA_PATH)
    # Assert
    assert fs.listdir(CLASSIFIED_DATA_PATH) == FILES
    _assert_file_contents_equal(CLASSIFIED_DATA_PATH, FILES, expected_file_contents)


def _create_and_write_raw_data(fs):
    """Setup raw data folder with 3 files of unclassified data"""
    fs.create_dir(RAW_DATA_PATH)
    raw_file_content = __create_raw_file_content(num_rows=3)
    for f in FILES:
        fs.create_file(f'{RAW_DATA_PATH}/{f}', contents=raw_file_content)


def __create_raw_file_content(num_rows=3):
    """Creates a string with the content for an unclassified data file"""
    file_content = "date,headline,topic\n"
    for i in range(num_rows):
        date = pd.Timestamp('2022-01-01') + pd.Timedelta(days=i)
        date = date.strftime('%Y-%m-%d 00:00:00')
        line = f"{date},headline,none\n"
        file_content += line
    return file_content


def _create_and_write_classified_data(fs):
    """Setup classified data folder with 2 files of classified data
    where one file is partially classified
    """
    fs.create_dir(CLASSIFIED_DATA_PATH)
    classified_file_content = __create_classified_file_content(num_rows=3)
    partially_classified_file_content = __create_classified_file_content(num_rows=2)

    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-01.csv',
                   contents=classified_file_content)
    fs.create_file(f'{CLASSIFIED_DATA_PATH}/2022-02.csv',
                   contents=partially_classified_file_content)
    return classified_file_content, partially_classified_file_content


def __create_classified_file_content(num_rows=3):
    """Creates a string with the content for a classified data file"""
    file_content = "date,headline,topic,model topic,sentiment\n"
    for i in range(num_rows):
        date = pd.Timestamp('2022-01-01') + pd.Timedelta(days=i)
        date = date.strftime('%Y-%m-%d 00:00:00')
        line = f"{date},headline,none,test,test\n"
        file_content += line
    return file_content


def _assert_file_contents_equal(data_path, files, file_contents):
    file_paths = [f'{data_path}/{f}' for f in files]
    for i in range(len(file_paths)):
        assert _fs_read_file(file_paths[i]) == file_contents[i]
