import pandas as pd

from constants import Constants
from dataset_models import Dataset
from dataset_models_bombay import DatasetBombay
from dataset_models_sdss import DatasetSDSS
from helper_dataset_creation import find_similar_queries, find_different_queries, create_siamese_batch_row, \
    create_triplet_batch_row


def create_and_save_sbert_datasets(dataset: Dataset):
    triplet_column_list = [Constants.COLUMN_NAME_ANCHOR, Constants.COLUMN_NAME_POSITIVE, Constants.COLUMN_NAME_NEGATIVE]
    siamese_column_list = [
        Constants.COLUMN_NAME_QUERY_ONE,
        Constants.COLUMN_NAME_QUERY_TWO,
        Constants.COLUMN_NAME_SIMILARITY_SCORE
    ]
    siamese_df = pd.DataFrame(columns=siamese_column_list)
    triplet_df = pd.DataFrame(columns=triplet_column_list)

    # Process and get dataset as a pandas dataframe
    # This processed dataset is supposed to have the following column names:
    # full_query, labels
    df = dataset.process()
    labeled_df = df[df.labels != -1][['full_query', 'labels']]

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

    siamese_df.to_csv(dataset.get_path_siamese_dataset(), index=False)
    triplet_df.to_csv(dataset.get_path_triplet_dataset(), index=False)

    print('Datasets are saved!')


if __name__ == '__main__':
    create_and_save_sbert_datasets(DatasetSDSS())
    create_and_save_sbert_datasets(DatasetBombay())
