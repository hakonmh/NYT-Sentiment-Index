from datetime import datetime
import pandas as pd
from src.download import (
    nyt_download_history,
    nyt_download_latest_month,
    __get_nyt_headlines,
    __format_headlines,
)


def test_nyt_download_history(mocker, fs):
    # Arrange
    dummy_data = [('2000-01-01', 'headline', 'topic')]
    mocker.patch('src.download.__get_nyt_headlines', return_value=dummy_data)
    # Act
    nyt_download_history(start_year=2000)
    # Assert
    _assert_raw_data_files_exists(fs, start_year=2000)


def _assert_raw_data_files_exists(fs, start_year=2000):
    for year in range(start_year, datetime.now().year + 1):
        for month in range(1, 13):
            if datetime(year, month, 1) > datetime.now():
                break
            assert fs.exists(f'data/raw-nyt-data/{year}-{month:02d}.csv')


def test_nyt_download_latest_month(mocker, fs):
    # Arrange
    year = datetime.now().year
    month = datetime.now().month
    dummy_data = [('2000-01-01', 'headline', 'topic')]
    mocker.patch('src.download.__get_nyt_headlines', return_value=dummy_data)
    # Act
    nyt_download_latest_month()
    # Assert
    assert fs.exists(f'data/raw-nyt-data/{year}-{month:02d}.csv')


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
