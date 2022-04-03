class Constants:
    URL_DATASET_PREPROCESSED_TOKENIZED = "https://raw.githubusercontent.com/ShirinTahmasebi/KTH-ID2223/main/dataset/preprocessed_tokenized_sdss.csv"

    DATASET_PATH_SIAMESE = '/home/shirin/query_embedding/df_siamese.csv'
    DATASET_PATH_TRIPLET = '/home/shirin/query_embedding/df_triplet.csv'

    BERT_MODEL_NAME_BASE_UNCASED = "bert-base-uncased"
    BERT_MODEL_NAME_CODE_BERT = "microsoft/codebert-base-mlm"

    PARAMETERS_EPOCHS = 1
    PARAMETERS_SPLIT_TRAIN_EVAL = .2
    PARAMETERS_BATCH_SIZE = 4

    COLUMN_NAME_QUERY_ONE = 'query_one'
    COLUMN_NAME_QUERY_TWO = 'query_two'
    COLUMN_NAME_SIMILARITY_SCORE = 'similarity_score'

    COLUMN_NAME_ANCHOR = 'anchor'
    COLUMN_NAME_POSITIVE = 'positive'
    COLUMN_NAME_NEGATIVE = 'negative'

    SIMILAR_QUERIES_COUNT = 10
    DIFFERENT_QUERIES_COUNT = 10