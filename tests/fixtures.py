RAW_DATA_PATH = 'data/raw-nyt-data'
CLASSIFIED_DATA_PATH = 'data/classified-nyt-data'


def _fs_read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
