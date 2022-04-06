class Constants:
    URL_DATASET_PREPROCESSED_TOKENIZED = "https://raw.githubusercontent.com/ShirinTahmasebi/KTH-ID2223/main/dataset/preprocessed_tokenized_sdss.csv"

    PATH_DATASET_TOKENIZED_AND_LABELED = '/home/shirin/query_embedding/df_labeled.csv'
    PATH_DATASET_SIAMESE = '/home/shirin/query_embedding/df_siamese.csv'
    PATH_DATASET_TRIPLET = '/home/shirin/query_embedding/df_triplet.csv'
    PATH_TOKENIZER = '/home/shirin/query_embedding/tokenizer'
    PATH_TOKENIZED_DF = '/home/shirin/query_embedding/tokenized_df'
    PATH_TOKENIZED_SIAMESE_ONE = '/home/shirin/query_embedding/tokenized_siamese_one'
    PATH_TOKENIZED_SIAMESE_TWO = '/home/shirin/query_embedding/tokenized_siamese_two'
    PATH_TOKENIZED_TRIPLET_ANCHOR = '/home/shirin/query_embedding/tokenized_triplet_anchor'
    PATH_TOKENIZED_TRIPLET_POSITIVE = '/home/shirin/query_embedding/tokenized_triplet_positive'
    PATH_TOKENIZED_TRIPLET_NEGATIVE = '/home/shirin/query_embedding/tokenized_triplet_negative'
    PATH_FINE_TUNED_MODEL_WEIGHTS_SIAMESE = '/home/shirin/query_embedding/bert_siamese_weights'
    PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET_BERT = '/home/shirin/query_embedding/bert_triple_weights'
    PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET_CODE_BERT = '/home/shirin/query_embedding/codebert_triple_weights'
    PATH_PREDICTED_TRIPLET_BERT_2D_X = '/home/shirin/query_embedding/results/{}/predicted_points_x_triplet_bert.pickle'
    PATH_PREDICTED_TRIPLET_BERT_2D_Y = '/home/shirin/query_embedding/results/{}/predicted_points_y_triplet_bert.pickle'
    PATH_PREDICTED_TRIPLET_BERT_LABELS = '/home/shirin/query_embedding/results/{}/predicted_points_labels_triplet_bert.pickle'
    PATH_PREDICTED_TRIPLET_BERT_CLS = '/home/shirin/query_embedding/results/{}/predicted_points_cls_list.pickle'
    PATH_PREDICTED_TRIPLET_CODE_BERT_2D_X = '/home/shirin/query_embedding/results/{}/code_bert_predicted_points_x_triplet_bert.pickle'
    PATH_PREDICTED_TRIPLET_CODE_BERT_2D_Y = '/home/shirin/query_embedding/results/{}/code_bert_predicted_points_y_triplet_bert.pickle'
    PATH_PREDICTED_TRIPLET_CODE_BERT_LABELS = '/home/shirin/query_embedding/results/{}/code_bert_predicted_points_labels_triplet_bert.pickle'
    PATH_PREDICTED_TRIPLET_CODE_BERT_CLS = '/home/shirin/query_embedding/results/{}/code_bert_predicted_points_cls_list.pickle'

    BERT_MODEL_NAME_BASE_UNCASED = "bert-base-uncased"
    BERT_MODEL_NAME_CODE_BERT = "microsoft/codebert-base-mlm"
    BERT_MODEL_NAME = BERT_MODEL_NAME_CODE_BERT

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

    MAX_LENGTH = 512

    TRIPLET_BERT_RESULT_PATH_DIC = {
        'x': PATH_PREDICTED_TRIPLET_BERT_2D_X,
        'y': PATH_PREDICTED_TRIPLET_BERT_2D_Y,
        'labels': PATH_PREDICTED_TRIPLET_BERT_LABELS,
        'cls': PATH_PREDICTED_TRIPLET_BERT_CLS
    }

    TRIPLET_CODE_BERT_RESULT_PATH_DIC = {
        'x': PATH_PREDICTED_TRIPLET_CODE_BERT_2D_X,
        'y': PATH_PREDICTED_TRIPLET_CODE_BERT_2D_Y,
        'labels': PATH_PREDICTED_TRIPLET_CODE_BERT_LABELS,
        'cls': PATH_PREDICTED_TRIPLET_CODE_BERT_CLS
    }
