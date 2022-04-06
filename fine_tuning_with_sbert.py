from helper import *
from helper_prediction import get_tokenized_siamese_inputs, load_datasets, get_tokenized_triplet_inputs, \
    train_triplet_model
from prediction import predict_and_save_results

# TODO Shirin: Needs refactoring!
if __name__ == '__main__':
    df, siamese_df, triplet_df = load_datasets(
        Constants.URL_DATASET_PREPROCESSED_TOKENIZED,
        Constants.PATH_DATASET_SIAMESE,
        Constants.PATH_DATASET_TRIPLET
    )

    sql_tokenizer = load_from_disk_or_call_func(Constants.PATH_TOKENIZER, initialize_sql_base_tokenizer, df)

    tokenized_input_siamese_one, tokenized_input_siamese_two = get_tokenized_siamese_inputs(siamese_df, sql_tokenizer)
    tokenized_triplet_inputs = get_tokenized_triplet_inputs(triplet_df, sql_tokenizer)

    # Train Triplet Model.
    trained_triplet_model = train_triplet_model(Constants.BERT_MODEL_NAME, *tokenized_triplet_inputs)
    trained_triplet_model.save_weights(Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TRIPLET_BERT)

    # Evaluate the trained Triplet model.
    predict_and_save_results(
        trained_triplet_model.layers[0].layers[9].layers[3],
        Constants.TRIPLET_BERT_RESULT_PATH_DIC
    )
    #
    # siamese_network = SBertModelSiamese(create_sbert_model_siamese())
    # siamese_network.compile(optimizer=optimizers.Adam(0.0001))
    # siamese_network.fit(
    #     [
    #         tf.convert_to_tensor(tokenized_input_siamese_one['input_ids'].numpy()),
    #         tf.convert_to_tensor(tokenized_input_siamese_one['attention_mask'].numpy()),
    #         tf.convert_to_tensor(tokenized_input_siamese_one['token_type_ids'].numpy()),
    #
    #         tf.convert_to_tensor(tokenized_input_siamese_two['input_ids'].numpy()),
    #         tf.convert_to_tensor(tokenized_input_siamese_two['attention_mask'].numpy()),
    #         tf.convert_to_tensor(tokenized_input_siamese_two['token_type_ids'].numpy())
    #     ],
    #     epochs=10,
    #     batch_size=4,
    #     steps_per_epoch=100,
    #     validation_steps=10
    # )
    #
    # siamese_network.save_weights(Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_SIAMESE)
