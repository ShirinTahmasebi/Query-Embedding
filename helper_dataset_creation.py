import pandas as pd
import numpy as np

from constants import Constants


def create_siamese_row(query_one: str, query_two: str, similarity_score: int):
    return pd.Series({
        Constants.COLUMN_NAME_QUERY_ONE: query_one,
        Constants.COLUMN_NAME_QUERY_TWO: query_two,
        Constants.COLUMN_NAME_SIMILARITY_SCORE: similarity_score
    })


def create_siamese_batch_row(query_base: str, query_two_series: pd.Series, similarity_score: int):
    output = pd.DataFrame(
        columns=[
            Constants.COLUMN_NAME_QUERY_ONE,
            Constants.COLUMN_NAME_QUERY_TWO,
            Constants.COLUMN_NAME_SIMILARITY_SCORE
        ])

    # Be aware that, here, the order of lines matters!
    output[Constants.COLUMN_NAME_QUERY_TWO] = query_two_series
    output[Constants.COLUMN_NAME_QUERY_ONE] = query_base
    output[Constants.COLUMN_NAME_SIMILARITY_SCORE] = similarity_score

    return output


def create_triplet_row(query_base: str, query_negative: str, query_positive: str):
    return pd.Series({
        Constants.COLUMN_NAME_ANCHOR: query_base,
        Constants.COLUMN_NAME_POSITIVE: query_positive,
        Constants.COLUMN_NAME_NEGATIVE: query_negative
    })


def create_triplet_batch_row(
        query_base: list,
        query_negative_series: pd.Series,
        query_positive_series: pd.Series
):
    output = pd.DataFrame(columns=[
        Constants.COLUMN_NAME_ANCHOR,
        Constants.COLUMN_NAME_POSITIVE,
        Constants.COLUMN_NAME_NEGATIVE
    ])

    # Be aware that, here, the order of lines matters!
    output[Constants.COLUMN_NAME_POSITIVE] = query_positive_series
    output[Constants.COLUMN_NAME_NEGATIVE] = query_negative_series
    output[Constants.COLUMN_NAME_ANCHOR] = query_base

    return output


def find_similar_queries(df: pd.DataFrame, query: str, label: int, similar_labels: list):
    all_similar_queries = df[(df.labels == label) & (df.full_query != query)]
    if len(all_similar_queries) < Constants.SIMILAR_QUERIES_COUNT:
        queries_with_identical_labels_count = len(all_similar_queries)
        all_similar_queries.append(df[df.labels.isin(similar_labels)])
        if queries_with_identical_labels_count == len(all_similar_queries):
            random_selection = list(range(queries_with_identical_labels_count))
        else:
            random_selection = \
                list(range(queries_with_identical_labels_count)) + \
                list(np.random.randint(low=queries_with_identical_labels_count, high=len(all_similar_queries) + 1,
                                       size=Constants.SIMILAR_QUERIES_COUNT - queries_with_identical_labels_count))
    else:
        random_selection = list(
            np.random.randint(low=0, high=len(all_similar_queries), size=Constants.SIMILAR_QUERIES_COUNT))

    similar_queries = all_similar_queries.iloc[random_selection]['full_query']
    return similar_queries.reset_index(drop=True)


def find_different_queries(df: pd.DataFrame, label: int, similar_labels: list):
    different_df = df[(df.labels != label) & (~df.labels.isin(similar_labels))]['full_query']
    random_selection = list(np.random.randint(low=0, high=len(different_df), size=Constants.DIFFERENT_QUERIES_COUNT))

    different_queries = different_df.iloc[random_selection]
    return different_queries.reset_index(drop=True)
