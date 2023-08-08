from newsindex import download
from newsindex import classify
from newsindex import create_index


def make_whole_history(start_year=1852):
    """Downloads, classifes and creates the sentiment index from scratch.

    Parameters
    ----------
    start_year: int
        The year from which to start downloading the data. Defaults to 1852.
    """
    download.nyt_download_history(start_year, overwrite=False)
    classify.classify_full_history(overwrite=False)
    create_index.create_sentiment_index()


def make_latest():
    """Download and appends latest data to existing index."""
    download.nyt_download_latest()
    classify.classify_latest()
    create_index.append_sentiment_index()


if __name__ == "__main__":
    # make_whole_history(start_year=1852)
    make_latest()
