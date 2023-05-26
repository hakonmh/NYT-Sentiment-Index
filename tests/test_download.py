import pytest
from .fixtures import _fs_read_file

from datetime import datetime
from src.download import (
    nyt_download_history,
    nyt_download_latest,
    _download_nyt_headlines_for_month,
    NYT_OUTPUT_PATH
)


def test_nyt_download_history(mocker, fs):
    # Arrange
    expected_files = _create_expected_listdir_content(start_year=2022)
    dummy_articles = _get_dummy_articles(num_rows=1)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=dummy_articles)
    # Act
    nyt_download_history(start_year=2022)
    # Assert
    assert fs.listdir(NYT_OUTPUT_PATH) == expected_files


def test_nyt_download_history_content(mocker, fs):
    # Arrange
    dummy_articles = _get_dummy_articles(num_rows=1)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=dummy_articles)
    expected_file_content = _create_expected_file_content(dummy_articles)
    # Act
    nyt_download_history(start_year=2022)
    # Assert
    assert _fs_read_file(f'{NYT_OUTPUT_PATH}/2022-01.csv') == expected_file_content


def test_nyt_download_history_overwrite(mocker, fs):
    # Arrange
    old_dummy_articles = _get_dummy_articles(num_rows=1)
    old_file_content = _create_expected_file_content(old_dummy_articles)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=old_dummy_articles)
    nyt_download_history(start_year=2022)

    new_dummy_articles = _get_dummy_articles(num_rows=2)
    expected_file_content = _create_expected_file_content(new_dummy_articles)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=new_dummy_articles)
    # Act
    nyt_download_history(start_year=2022, overwrite=True)
    # Assert
    assert _fs_read_file(f'{NYT_OUTPUT_PATH}/2022-01.csv') != old_file_content
    assert _fs_read_file(f'{NYT_OUTPUT_PATH}/2022-01.csv') == expected_file_content


def test_nyt_download_history_not_overwrite(mocker, fs):
    # Arrange
    old_dummy_articles = _get_dummy_articles(num_rows=1)
    expected_file_content = _create_expected_file_content(old_dummy_articles)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=old_dummy_articles)
    nyt_download_history(start_year=2022)

    new_dummy_articles = _get_dummy_articles(num_rows=2)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=new_dummy_articles)
    # Act & Assert
    with pytest.warns(UserWarning):
        nyt_download_history(start_year=2022, overwrite=False, warn=True)
    assert _fs_read_file(f'{NYT_OUTPUT_PATH}/2022-01.csv') == expected_file_content


def test_nyt_download_latest(mocker, fs):
    # Arrange
    fs.create_dir(NYT_OUTPUT_PATH)

    dummy_articles = _get_dummy_articles(num_rows=1)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=dummy_articles)
    mocker.patch('os.listdir', return_value=['2022-01-01.csv'])
    expected_files = _create_expected_listdir_content(start_year=2022)
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


def test_download_nyt_headlines_for_month(mocker, fs):
    # Arrange
    fs.makedirs(NYT_OUTPUT_PATH)
    dummy_articles = _get_dummy_articles(num_rows=2)
    mocker.patch('pynytimes.NYTAPI.archive_metadata', return_value=dummy_articles)
    expected_file_content = _create_expected_file_content(dummy_articles)
    # Act
    _download_nyt_headlines_for_month(month=1, year=2022, output_folder=NYT_OUTPUT_PATH)
    # Assert
    assert _fs_read_file(f'{NYT_OUTPUT_PATH}/2022-01.csv') == expected_file_content


def _get_dummy_articles(num_rows=1):
    dummy_articles = []
    for i in range(1, num_rows + 1):
        article = {
            "pub_date": f"2000-01-{i:02d}",
            "headline": {"main": f"headline{i}"},
            "news_desk": f"topic{i}"
        }
        dummy_articles.append(article)
    return dummy_articles


def _create_expected_file_content(dummy_articles):
    expected_file_content = 'date,headline,topic\n'
    for article in dummy_articles:
        expected_file_content += f"{article['pub_date']},"\
                                 f"{article['headline']['main']},"\
                                 f"{article['news_desk']}\n"
    return expected_file_content
