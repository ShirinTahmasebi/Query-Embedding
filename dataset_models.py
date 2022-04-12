import abc

import pandas as pd
from pandas import DataFrame


class Dataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_labeled(self):
        pass

    @abc.abstractmethod
    def get_path_csv(self):
        pass

    def get_delimiter(self):
        return ','

    @abc.abstractmethod
    def get_path_labeled_dataset(self):
        pass

    @abc.abstractmethod
    def get_path_siamese_dataset(self):
        pass

    @abc.abstractmethod
    def get_path_triplet_dataset(self):
        pass

    @abc.abstractmethod
    def add_label(self, df: DataFrame):
        pass

    @abc.abstractmethod
    def get_similar_labels(self, label: int):
        return []

    @abc.abstractmethod
    def preprocess(self, df: DataFrame):
        return df

    @abc.abstractmethod
    def postprocess(self, df: DataFrame):
        return df

    def process(self):
        df = pd.read_csv(self.get_path_csv(), self.get_delimiter())
        df = self.preprocess(df)

        if not self.is_labeled():
            labeled_df = self.add_label(df)
            labeled_df.to_csv(self.get_path_labeled_dataset(), index=False)

        df = self.postprocess(df)
        return df[df.labels != -1]
