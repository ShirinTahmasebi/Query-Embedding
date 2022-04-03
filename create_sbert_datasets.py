import pandas as pd
import numpy as np

from constants import Constants
from query_pattern_regex import *
from query_pattern import QueryPatterns, QueryPattern


def create_siamese_row(query_tokens_one: list, query_tokens_two: list, similariry_score: int):
    return pd.Series({
        Constants.COLUMN_NAME_QUERY_ONE: query_tokens_one,
        Constants.COLUMN_NAME_QUERY_TWO: query_tokens_two,
        Constants.COLUMN_NAME_SIMILARITY_SCORE: similariry_score
    })


def create_siamese_batch_row(query_tokens_base: list, query_tokens_two_list: list, similarity_score: int):
    output = pd.DataFrame(columns=[Constants.COLUMN_NAME_QUERY_ONE, Constants.COLUMN_NAME_QUERY_TWO,
                                   Constants.COLUMN_NAME_SIMILARITY_SCORE])
    for query_two in query_tokens_two_list:
        output = output.append({
            Constants.COLUMN_NAME_QUERY_ONE: query_tokens_base,
            Constants.COLUMN_NAME_QUERY_TWO: query_two,
            Constants.COLUMN_NAME_SIMILARITY_SCORE: similarity_score
        }, ignore_index=True)
    return output


def create_triplet_row(query_tokens_base: list, query_tokens_negative: list, query_tokens_positive: int):
    return pd.Series({
        Constants.COLUMN_NAME_ANCHOR: query_tokens_base,
        Constants.COLUMN_NAME_POSITIVE: query_tokens_positive,
        Constants.COLUMN_NAME_NEGATIVE: query_tokens_negative
    })


def create_triplet_batch_row(
        query_tokens_base: list,
        query_tokens_negative_list: pd.Series,
        query_tokens_positive_list: pd.Series
):
    output = pd.DataFrame(columns=[
        Constants.COLUMN_NAME_ANCHOR,
        Constants.COLUMN_NAME_POSITIVE,
        Constants.COLUMN_NAME_NEGATIVE
    ])
    rows_count = min(len(query_tokens_negative_list), len(query_tokens_positive_list))

    for i in range(rows_count):
        output = output.append({
            Constants.COLUMN_NAME_ANCHOR: query_tokens_base,
            Constants.COLUMN_NAME_POSITIVE: query_tokens_positive_list.iloc[i],
            Constants.COLUMN_NAME_NEGATIVE: query_tokens_negative_list.iloc[i]
        }, ignore_index=True)

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

    similar_queries = all_similar_queries.iloc[random_selection]['tokens']
    return similar_queries


def find_different_queries(df: pd.DataFrame, label: int, similar_labels: list):
    different_df = df[(df.labels != label) & (~df.labels.isin(similar_labels))]['tokens']
    random_selection = list(np.random.randint(low=0, high=len(different_df), size=Constants.DIFFERENT_QUERIES_COUNT))

    different_queries = different_df.iloc[random_selection]
    return different_queries


if __name__ == '__main__':
    df = pd.read_csv(Constants.URL_DATASET_PREPROCESSED_TOKENIZED)

    patterns = QueryPatterns()
    patterns.add_item(QueryPattern(0, query_pattern_regex[0], []))
    patterns.add_item(QueryPattern(1, query_pattern_regex[1], []))
    patterns.add_item(QueryPattern(2, query_pattern_regex[2], []))
    patterns.add_item(QueryPattern(3, query_pattern_regex[3], []))
    patterns.add_item(QueryPattern(4, query_pattern_regex[4], [4, 5]))
    patterns.add_item(QueryPattern(5, query_pattern_regex[5], [4, 5]))
    patterns.add_item(QueryPattern(6, query_pattern_regex[6], [6, 7, 8, 9]))
    patterns.add_item(QueryPattern(7, query_pattern_regex[7], [6, 7, 8, 9]))
    patterns.add_item(QueryPattern(8, query_pattern_regex[8], [6, 7, 8, 9]))
    patterns.add_item(QueryPattern(9, query_pattern_regex[9], [6, 7, 8, 9]))
    patterns.add_item(QueryPattern(10, query_pattern_regex[10], []))
    patterns.add_item(QueryPattern(11, query_pattern_regex[11], []))

    df['labels'] = df.apply(lambda x: patterns.get_matched_pattern_key(x.full_query), axis=1)
    labeled_df = df[df.labels != -1][['full_query', 'tokens', 'labels']]

    similar_queries = []
    siamese_df = pd.DataFrame(columns=[Constants.COLUMN_NAME_QUERY_ONE, Constants.COLUMN_NAME_QUERY_TWO,
                                       Constants.COLUMN_NAME_SIMILARITY_SCORE])
    triplet_df = pd.DataFrame(
        columns=[Constants.COLUMN_NAME_ANCHOR, Constants.COLUMN_NAME_POSITIVE, Constants.COLUMN_NAME_NEGATIVE])

    for labeled_df_index, labeled_df_row in labeled_df.iterrows():
        query = labeled_df_row.full_query
        tokens = labeled_df_row.tokens
        label = labeled_df_row.labels
        similar_labels = patterns.get_similar_patterns(label)

        similar_queries = find_similar_queries(labeled_df, query, label, similar_labels)
        different_queries = find_different_queries(labeled_df, query, similar_labels)

        siamese_df = siamese_df.append(create_siamese_batch_row(tokens, similar_queries, 1))
        siamese_df = siamese_df.append(create_siamese_batch_row(tokens, different_queries, 0))

        triplet_df = triplet_df.append(create_triplet_batch_row(tokens, different_queries, similar_queries))

    siamese_df.to_csv(Constants.DATASET_PATH_SIAMESE, index=False)
    triplet_df.to_csv(Constants.DATASET_PATH_TRIPLET, index=False)