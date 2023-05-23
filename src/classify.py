import os
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import pipeline


class ClassificationPipeline:

    def __init__(self, batch_size=64, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.topic_model = _Model(
            "hakonmh/topic-xdistil-uncased",
            batch_size=batch_size,
            device=device
        )
        self.sentiment_model = _Model(
            "hakonmh/sentiment-xdistil-uncased",
            batch_size=batch_size,
            device=device
        )

    def predict(self, data):
        dataset = TextDataset(data)
        # ignore user warnings from transformers
        warnings.filterwarnings('ignore')
        topics = self.topic_model.predict(dataset)
        sentiments = self.sentiment_model.predict(dataset)
        warnings.resetwarnings()

        if isinstance(data, pd.DataFrame):
            df = data.copy()
            df['model topic'] = topics
            df['sentiment'] = sentiments
        else:
            df = pd.DataFrame(
                {'headline': data.texts,
                 'model topic': topics,
                 'sentiment': sentiments}
            )
            df.index = data.index
            df['headline'] = df['headline'].astype(str)
        return df


class TextDataset(Dataset):
    def __init__(self, texts):
        if isinstance(texts, str):
            self.texts = [texts]
            self.index = range(1)
        elif isinstance(texts, pd.Series):
            self.texts = texts.tolist()
            self.index = texts.index
        elif isinstance(texts, pd.DataFrame):
            self.texts = texts['headline'].tolist()
            self.index = texts.index
        else:
            self.texts = list(texts)
            self.index = range(len(texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


class _Model:
    def __init__(self, model_name, batch_size=64, device=None):
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            batch_size=batch_size,
            device=device,
            max_length=512,
            truncation=True
        )

    def predict(self, texts):
        return [pred['label'] for pred in self.pipeline(texts)]


def classify_full_history(
    input_folder='data/raw-nyt-data',
    output_folder='data/classified-nyt-data',
    overwrite=False
):
    os.makedirs(output_folder, exist_ok=True)

    pipeline = ClassificationPipeline()
    for f in _files_to_classify(input_folder, output_folder, overwrite):
        df = pd.read_csv(os.path.join(input_folder, f), index_col=0)
        df['headline'] = df['headline'].astype(str)
        df = df.dropna(subset=['headline'])
        classified_df = pipeline.predict(df)
        classified_df.to_csv(os.path.join(output_folder, f))


def _files_to_classify(input_folder, output_folder, overwrite):
    if overwrite:
        files = os.listdir(input_folder)
    else:
        # get files in input_folder that are not in output_folder
        input_files = os.listdir(input_folder)
        output_files = os.listdir(output_folder)
        files = list(set(input_files) - set(output_files))
        files = sorted(files)
    return [f for f in files if f.endswith(".csv")]


def classify_latest(
    input_folder='data/raw-nyt-data',
    output_folder='data/classified-nyt-data'
):
    pipeline = ClassificationPipeline()
    for f in _files_not_classified(input_folder, output_folder):
        df = pd.read_csv(os.path.join(input_folder, f), index_col=0)
        df['headline'] = df['headline'].astype(str)
        df = df.dropna(subset=['headline'])
        classified_df = pipeline.predict(df)
        classified_df.to_csv(os.path.join(output_folder, f))


def _files_not_classified(input_folder, output_folder):
    files = _files_to_classify(input_folder, output_folder, overwrite=False)
    # Re-classify latest file in case the raw data has been updated since last classification
    latest_classified_file = os.listdir(output_folder)[-1]
    files.insert(0, latest_classified_file)
    return files
