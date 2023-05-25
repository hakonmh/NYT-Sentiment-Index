import os
import sys
import warnings
import pandas as pd
from torch import cuda, device as _device
from torch.utils.data import Dataset
if 'pytest' not in sys.modules:
    from transformers import pipeline


class ClassificationPipeline:
    def __init__(self, batch_size=64, device=None):
        """A class for classifying the topic and sentiment of text data.

        Parameters
        ----------
        batch_size : int, optional
            The batch size to use for the models. Default is 64.
        device : str or torch.device, optional
            The device on which to run the models. If not provided, will default to GPU if available, else CPU.
        """
        if device is None:
            device = _device("cuda:0" if cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = _device(device)

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
        """Predict the topics and sentiments of the given text data.

        Parameters
        ----------
        data : str or Sequence[str]
            The text data to be classified.

        Returns
        -------
        DataFrame
            DataFrame with columns for the classified topics and sentiments.
        """
        dataset = TextDataset(data)
        topics = self.topic_model.predict(dataset)
        sentiments = self.sentiment_model.predict(dataset)

        if isinstance(data, pd.DataFrame):
            df = data.copy()
            df['model topic'] = topics
            df['sentiment'] = sentiments
        else:
            df = pd.DataFrame(
                {'headline': dataset.texts,
                 'model topic': topics,
                 'sentiment': sentiments}
            )
            df.index = dataset.index
            df['headline'] = df['headline'].astype(str)
        return df


class TextDataset(Dataset):
    def __init__(self, texts):
        """A PyTorch Dataset for handling a sequence of text.

        Parameters
        ----------
        texts : str or Sequence[str]
            The text data to be classified.
        """
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
        """A class for a classification model pipeline.

        Parameters
        ----------
        model_name : str
            The name of the model to use. Must be a model from the HuggingFace model hub.
        batch_size : int, optional
            The batch size to use for the models. Default is 64.
        device : str or torch.device, optional
            The device on which to run the models. Default is GPU if available.
        """
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
        """Predict the classifications of the given texts.

        Parameters
        ----------
        texts : list
            The texts to be classified.

        Returns
        -------
        list
            The predicted classifications of the texts.
        """
        warnings.filterwarnings('ignore')
        predictions = [pred['label'] for pred in self.pipeline(texts)]
        warnings.resetwarnings()
        return predictions


def classify_full_history(
    input_folder='data/raw-nyt-data',
    output_folder='data/classified-nyt-data',
    overwrite=False
):
    """Classify the sentiment and topic of headlines for all files in a given folder.

    Parameters
    ----------
    input_folder : str, optional
        Path to the folder containing the input files.
    output_folder : str, optional
        Path to the folder where the output files will be stored.
    overwrite : bool, optional
        Whether to overwrite existing output files.
    """
    os.makedirs(output_folder, exist_ok=True)

    pipeline = ClassificationPipeline()
    for f in _files_to_classify(input_folder, output_folder, overwrite):
        df = pd.read_csv(os.path.join(input_folder, f), index_col=0)
        df['headline'] = df['headline'].astype(str)
        df = df.dropna(subset=['headline'])
        classified_df = pipeline.predict(df)
        classified_df.to_csv(os.path.join(output_folder, f))


def _files_to_classify(input_folder, output_folder, overwrite):
    """Get the names of files to be classified.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the input files.
    output_folder : str
        Path to the folder where the output files are or will be stored.
    overwrite : bool
        Whether to overwrite existing output files.

    Returns
    -------
    list
        The names of the files to be classified.
    """
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
    """Classify the sentiment and topic of headlines for data in the input folder
    that has not been classified yet.

    Parameters
    ----------
    input_folder : str, optional
        Path to the folder containing the input files.
    output_folder : str, optional
        Path to the folder where the output files will be stored.
    """
    pipeline = ClassificationPipeline()
    for f in _files_not_classified(input_folder, output_folder):
        df = pd.read_csv(os.path.join(input_folder, f), index_col=0)
        df['headline'] = df['headline'].astype(str)
        df = df.dropna(subset=['headline'])
        classified_df = pipeline.predict(df)
        classified_df.to_csv(os.path.join(output_folder, f))


def _files_not_classified(input_folder, output_folder):
    """Get the names of files not yet classified.

    Includes the file with the latest date in the input_folder since it might contain
    unclassified data if the raw data has been updated since last classification.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the input files.
    output_folder : str
        Path to the folder where the output files are or will be stored.

    Returns
    """
    files = _files_to_classify(input_folder, output_folder, overwrite=False)
    # Re-classify latest file in case the raw data has been updated since last classification
    latest_classified_file = os.listdir(output_folder)[-1]
    files.insert(0, latest_classified_file)
    return files
