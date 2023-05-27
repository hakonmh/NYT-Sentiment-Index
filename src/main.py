import download
import classify
import create_index


def make_whole_history():
    download.nyt_download_history(1852, overwrite=False)
    classify.classify_full_history(overwrite=False)
    create_index.create_sentiment_index()


def make_latest():
    download.nyt_download_latest()
    classify.classify_latest()
    create_index.append_sentiment_index()


if __name__ == "__main__":
    make_whole_history()
    make_latest()
