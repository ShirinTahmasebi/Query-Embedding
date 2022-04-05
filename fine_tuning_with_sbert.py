import tensorflow as tf

from helper import *
from sbert_model_creator import create_sbert_model_triplet, create_sbert_model_siamese
from sbert_model_siamese import SBertModelSiamese
from sbert_model_triplet import SBertModelTriplet


def load_from_disk_or_call_func(path: str, function, *args):
    import pickle
    try:
        loaded_object = pickle.load(open(path, 'rb'))
        if loaded_object:
            return loaded_object
    except Exception:
        print("File ", path, " did not exist. Start creating the file ...")

    calculated_object = function(*args)
    pickle.dump(calculated_object, open(path, 'wb'))
    return calculated_object


# TODO Shirin: Needs refactoring!
if __name__ == '__main__':
    df = pd.read_csv(Constants.URL_DATASET_PREPROCESSED_TOKENIZED)
    siamese_df = pd.read_csv(Constants.PATH_DATASET_SIAMESE)
    triplet_df = pd.read_csv(Constants.PATH_DATASET_TRIPLET)

    sql_tokenizer = load_from_disk_or_call_func(Constants.PATH_TOKENIZER, initialize_sql_base_tokenizer, df)
    # TODO Shirin: Fix this inconsistency!
    tokenized_input_siamese_one = load_from_disk_or_call_func(
        Constants.PATH_TOKENIZED_SIAMESE_ONE, sql_based_tokenizer, siamese_df['query_one'], sql_tokenizer
    )

    tokenized_input_siamese_two = load_from_disk_or_call_func(
        Constants.PATH_TOKENIZED_SIAMESE_TWO,
        sql_based_tokenizer,
        siamese_df['query_two'],  # This should be passed to the 'sql_based_tokenizer' function.
        sql_tokenizer  # This should be passed to the 'sql_based_tokenizer' function.
    )[1]

    tokenized_input_triplet_anchor = load_from_disk_or_call_func(
        Constants.PATH_TOKENIZED_TRIPLET_ANCHOR,
        sql_based_tokenizer,
        triplet_df['anchor'],  # This should be passed to the 'sql_based_tokenizer' function.
        sql_tokenizer  # This should be passed to the 'sql_based_tokenizer' function.
    )[1]

    tokenized_input_triplet_positive = load_from_disk_or_call_func(
        Constants.PATH_TOKENIZED_TRIPLET_POSITIVE,
        sql_based_tokenizer,
        triplet_df['positive'],  # This should be passed to the 'sql_based_tokenizer' function.
        sql_tokenizer  # This should be passed to the 'sql_based_tokenizer' function.
    )[1]

    tokenized_input_triplet_negative = load_from_disk_or_call_func(
        Constants.PATH_TOKENIZED_TRIPLET_NEGATIVE,
        sql_based_tokenizer,
        triplet_df['negative'],  # This should be passed to the 'sql_based_tokenizer' function.
        sql_tokenizer  # This should be passed to the 'sql_based_tokenizer' function.
    )[1]

    from tensorflow.keras import optimizers

    triplet_network = SBertModelTriplet(create_sbert_model_triplet())
    triplet_network.compile(optimizer=optimizers.Adam(0.0001))
    triplet_network.fit(
        [
            tf.convert_to_tensor(tokenized_input_triplet_anchor['input_ids'].numpy()),
            tf.convert_to_tensor(tokenized_input_triplet_anchor['attention_mask'].numpy()),
            tf.convert_to_tensor(tokenized_input_triplet_anchor['token_type_ids'].numpy()),

            tf.convert_to_tensor(tokenized_input_triplet_positive['input_ids'].numpy()),
            tf.convert_to_tensor(tokenized_input_triplet_positive['attention_mask'].numpy()),
            tf.convert_to_tensor(tokenized_input_triplet_positive['token_type_ids'].numpy()),

            tf.convert_to_tensor(tokenized_input_triplet_negative['input_ids'].numpy()),
            tf.convert_to_tensor(tokenized_input_triplet_negative['attention_mask'].numpy()),
            tf.convert_to_tensor(tokenized_input_triplet_negative['token_type_ids'].numpy())
        ],
        epochs=10,
        batch_size=4,
        steps_per_epoch=100,
        validation_steps=10
    )

    triplet_network.save_weights(Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET)

    siamese_network = SBertModelSiamese(create_sbert_model_siamese())
    siamese_network.compile(optimizer=optimizers.Adam(0.0001))
    siamese_network.fit(
        [
            tf.convert_to_tensor(tokenized_input_siamese_one['input_ids'].numpy()),
            tf.convert_to_tensor(tokenized_input_siamese_one['attention_mask'].numpy()),
            tf.convert_to_tensor(tokenized_input_siamese_one['token_type_ids'].numpy()),

            tf.convert_to_tensor(tokenized_input_siamese_two['input_ids'].numpy()),
            tf.convert_to_tensor(tokenized_input_siamese_two['attention_mask'].numpy()),
            tf.convert_to_tensor(tokenized_input_siamese_two['token_type_ids'].numpy())
        ],
        epochs=10,
        batch_size=4,
        steps_per_epoch=100,
        validation_steps=10
    )

    siamese_network.save_weights(Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_SIAMESE)
