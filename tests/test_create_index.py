from .fixtures import CLASSIFIED_DATA_PATH

import re
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import numpy as np

from src.create_index import *

EXPECTED_COLUMNS = ['negative', 'neutral', 'positive',
                    'total', 'index_value', 'smoothed_index_value']
EXPECTED_LOC_ROW = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, np.nan]).values


def test_create_sentiment_index(fs, mocker):
    """Test creating sentiment index."""
    # Arrange
    _setup_fs(fs)
    # Act
    result_df = create_sentiment_index()
    row = result_df.loc['2022-02-07'].values
    # Assert
    assert result_df.columns.tolist() == EXPECTED_COLUMNS
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert np.allclose(row, EXPECTED_LOC_ROW, equal_nan=True)
    assert result_df.shape == (18, 6)


def test_append_sentiment_index(fs, mocker):
    """Test appending new data to sentiment index."""
    # Arrange
    _setup_fs(fs)
    old_index_df = _write_index_to_be_appended()
    # Act
    result_df = append_sentiment_index()
    row = result_df.loc['2022-02-07'].values
    # Assert
    assert result_df.loc[old_index_df.index].equals(old_index_df)
    assert result_df.shape == (18, 6)
    assert np.allclose(row, EXPECTED_LOC_ROW, equal_nan=True)


def _setup_fs(fs, num_files=2, num_rows=10):
    """Setup fake filesystem with some files of unclassified data in raw-data dir"""
    fs.create_dir(CLASSIFIED_DATA_PATH)

    for month in range(1, num_files + 1):
        file_content = _create_classified_file_content(num_rows, month=month)
        file_name = f'2022-{month:02d}.csv'
        fs.create_file(f'{CLASSIFIED_DATA_PATH}/{file_name}', contents=file_content)


def _create_classified_file_content(num_rows=30, month=1):
    """Creates a string with the content for a classified data file"""
    dates = __create_dates_list(num_rows, month=month)
    possible_sentiments = ['Positive', 'Negative', 'Neutral']
    sentiments = [possible_sentiments[i % 3] for i in range(num_rows)]
    possible_model_topics = ['Economics', 'Other']
    model_topics = [possible_model_topics[i % 2] for i in range(num_rows)]

    file_content = "date,headline,topic,model topic,sentiment\n"
    for i in range(num_rows):
        date = dates[i]
        headline = f'A headline {i}'
        topic = 'none'
        model_topic = model_topics[i]
        sentiment = sentiments[i]

        line = ','.join([date, headline, topic, model_topic, sentiment]) + '\n'
        file_content += line
    return file_content


def __create_dates_list(num_rows, month=1):
    """Creates a list of dates with a step size changing between 12 hours and 36 hours
    to simulate some dates missing headlines, while others days have multiple headlines
    """
    dates = []
    current_date = datetime(2022, month, 1, 0, 0, 0)
    step_sizes = [timedelta(hours=12), timedelta(hours=36)]
    for i in range(num_rows):
        dates.append(current_date.strftime('%Y-%m-%d %H:%M:%S'))
        step_size = step_sizes[i % 2]
        current_date += step_size
    return dates


def _write_index_to_be_appended():
    """Write index to file and return the index as a dataframe"""
    text = """date,negative,neutral,positive,total,index_value,smoothed_index_value
    2022-01-01,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-02,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-03,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-04,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-05,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-06,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-07,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-08,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-01-09,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-02-01,0.0,0.0,5.0,5.0,1.000000,NaN
    2022-02-02,0.0,0.0,5.0,5.0,1.000000,NaN
    """
    text = re.sub(r'\s+', '\n', text)
    index_df = pd.read_csv(StringIO(text), index_col=0, parse_dates=True)
    index_df.to_csv('data/nyt-index.csv')
    return index_df
