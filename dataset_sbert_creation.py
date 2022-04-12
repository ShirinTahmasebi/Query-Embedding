import pandas as pd

from constants import Constants
from dataset_models import DatasetSDSS
from helper_dataset_creation import find_similar_queries, find_different_queries, create_siamese_batch_row, \
    create_triplet_batch_row

if __name__ == '__main__':
    triplet_column_list = [Constants.COLUMN_NAME_ANCHOR, Constants.COLUMN_NAME_POSITIVE, Constants.COLUMN_NAME_NEGATIVE]
    simese_column_list = [
        Constants.COLUMN_NAME_QUERY_ONE,
        Constants.COLUMN_NAME_QUERY_TWO,
        Constants.COLUMN_NAME_SIMILARITY_SCORE
    ]

    dataset = DatasetSDSS()
    df = dataset.process()
    labeled_df = df[df.labels != -1][['full_query', 'tokens', 'labels']]

    similar_queries = []
    siamese_df = pd.DataFrame(columns=simese_column_list)
    triplet_df = pd.DataFrame(columns=triplet_column_list)

    for labeled_df_index, labeled_df_row in labeled_df.iterrows():
        query = labeled_df_row.full_query
        label = labeled_df_row.labels
        similar_labels = dataset.get_similar_labels(label)

        similar_queries = find_similar_queries(labeled_df, query, label, similar_labels)
        different_queries = find_different_queries(labeled_df, query, similar_labels)

        siamese_df = siamese_df.append(create_siamese_batch_row(query, similar_queries, 1))
        siamese_df = siamese_df.append(create_siamese_batch_row(query, different_queries, 0))

        triplet_df = triplet_df.append(create_triplet_batch_row(query, different_queries, similar_queries))

    assert not siamese_df.isnull().values.any()
    assert not triplet_df.isnull().values.any()

    siamese_df.to_csv(Constants.PATH_DATASET_SIAMESE_SDSS, index=False)
    triplet_df.to_csv(Constants.PATH_DATASET_TRIPLET_SDSS, index=False)

    print('Datasets are saved!')
