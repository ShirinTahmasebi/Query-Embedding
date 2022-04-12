import abc

import pandas as pd
from pandas import DataFrame

from constants import Constants
from query_pattern import QueryPatterns, QueryPattern
from query_pattern_regex import *


class Dataset(metaclass=abc.ABCMeta):
    @property
    def is_labeled(self):
        return self.is_labeled

    @is_labeled.setter
    @abc.abstractmethod
    def is_labeled(self, value: bool):
        pass

    @abc.abstractmethod
    def get_path_csv(self):
        pass

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
        df = pd.read_csv(self.get_path_csv())
        df = self.preprocess(df)

        if not self.is_labeled:
            labeled_df = self.add_label(df)
            labeled_df.to_csv(self.get_path_labeled_dataset(), index=False)

        df = self.postprocess(df)
        return df


class DatasetSDSS(Dataset):

    def __init__(self):
        self.patterns = QueryPatterns()
        self.patterns.add_item(QueryPattern(0, query_pattern_regex[0], []))
        self.patterns.add_item(QueryPattern(1, query_pattern_regex[1], []))
        self.patterns.add_item(QueryPattern(2, query_pattern_regex[2], []))
        self.patterns.add_item(QueryPattern(3, query_pattern_regex[3], []))
        self.patterns.add_item(QueryPattern(4, query_pattern_regex[4], [4, 5]))
        self.patterns.add_item(QueryPattern(5, query_pattern_regex[5], [4, 5]))
        self.patterns.add_item(QueryPattern(6, query_pattern_regex[6], [6, 7, 8, 9]))
        self.patterns.add_item(QueryPattern(7, query_pattern_regex[7], [6, 7, 8, 9]))
        self.patterns.add_item(QueryPattern(8, query_pattern_regex[8], [6, 7, 8, 9]))
        self.patterns.add_item(QueryPattern(9, query_pattern_regex[9], [6, 7, 8, 9]))
        self.patterns.add_item(QueryPattern(10, query_pattern_regex[10], []))
        self.patterns.add_item(QueryPattern(11, query_pattern_regex[11], []))

    def get_similar_labels(self, label: int):
        return self.patterns.get_similar_patterns(label)

    def preprocess(self, df: DataFrame):
        return df

    def postprocess(self, df: DataFrame):
        return df

    def get_path_csv(self):
        return Constants.URL_DATASET_PREPROCESSED_TOKENIZED_SDSS

    def get_path_labeled_dataset(self):
        return Constants.PATH_DATASET_TOKENIZED_AND_LABELED_SDSS

    def get_path_siamese_dataset(self):
        return Constants.PATH_DATASET_SIAMESE_SDSS

    def get_path_triplet_dataset(self):
        return Constants.PATH_DATASET_TRIPLET_SDSS

    def add_label(self, df):
        df['labels'] = df.apply(lambda x: self.patterns.get_matched_pattern_key(x.full_query), axis=1)
        return df
