from datetime import datetime
import pandas as pd
from src.download import (
    nyt_download_history,
    nyt_download_latest,
    __get_nyt_headlines,
    __format_headlines,
    NYT_OUTPUT_PATH
)


def test_nyt_download_history(mocker, fs):
    # Arrange
    expected_files = _create_expected_listdir_content(start_year=2022)
    dummy_data = [('2022-01-01', 'headline', 'topic')]
    mocker.patch('src.download.__get_nyt_headlines', return_value=dummy_data)
    # Act
    nyt_download_history(start_year=2022)
    # Assert
    assert fs.listdir(NYT_OUTPUT_PATH) == expected_files


def test_nyt_download_latest_month(mocker, fs):
    # Arrange
    expected_files = _create_expected_listdir_content(start_year=2022)

    fs.create_dir(NYT_OUTPUT_PATH)
    mocker.patch('os.listdir', return_value=['2022-01-01.csv'])
    dummy_data = [('2022-01-01', 'headline', 'topic')]
    mocker.patch('src.download.__get_nyt_headlines', return_value=dummy_data)
    # Act
    nyt_download_latest()
    # Assert
    assert fs.listdir(NYT_OUTPUT_PATH) == expected_files


def _create_expected_listdir_content(start_year=2022):
    year = datetime.now().year
    month = datetime.now().month
    expected_files = []
    for year in range(start_year, datetime.now().year + 1):
        for month in range(1, 13):
            if datetime(year, month, 1) > datetime.now():
                break
            expected_files.append(f'{year}-{month:02d}.csv')
    return expected_files


def test_get_nyt_headlines(mocker):
    # Arrange
    dummy_articles = [{"pub_date": "2000-01-01",
                       "headline": {"main": "headline"},
                       "news_desk": "topic"}]
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=dummy_articles)
    date = datetime(2000, 1, 1)
    # Act
    data = __get_nyt_headlines(date)
    # Assert
    assert data == [('2000-01-01', 'headline', 'topic')]


def test_format_headlines():
    # Arrange
    data = [('2000-01-01', 'headline', 'topic'), ('2000-01-01', 'headline2', 'topic2')]
    expected = pd.DataFrame(data, columns=["date", "headline", "topic"])
    expected = expected.set_index("date")
    expected.index = pd.DatetimeIndex(expected.index)
    # Act
    df = __format_headlines(data)
    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.equals(expected)
