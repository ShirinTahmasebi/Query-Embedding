from pandas import DataFrame

from constants import Constants
from dataset_models import Dataset


class DatasetBombay(Dataset):
    def is_labeled(self):
        return True

    def get_path_csv(self):
        return Constants.URL_DATASET_PREPROCESSED_RAW_BOMBAY

    def get_delimiter(self):
        return '\t'

    def get_path_labeled_dataset(self):
        pass

    def get_path_siamese_dataset(self):
        return Constants.PATH_DATASET_SIAMESE_BOMBAY

    def get_path_triplet_dataset(self):
        return Constants.PATH_DATASET_TRIPLET_BOMBAY

    def add_label(self, df: DataFrame):
        pass

    def get_similar_labels(self, label: int):
        return []

    def preprocess(self, df: DataFrame):
        df = df[['label', 'query']]
        df.rename(columns={'label': 'labels', 'query': 'full_query'}, inplace=True)
        return df

    def postprocess(self, df: DataFrame):
        return df
