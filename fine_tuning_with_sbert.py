from helper import *
from helper_enum import AllModelTaskItems
from helper_prediction import *
from prediction import predict_and_save_results


def fine_tune_save_and_predict_triplet(model_task_item: ModelTaskItem, model_inputs):
    # Train Triplet Model.
    trained_triplet_model = train_triplet_model(model_task_item.get_model_name(), *model_inputs)
    trained_triplet_model.save_weights(model_task_item.get_weights_path())

    # Evaluate the trained Triplet model.
    predict_and_save_results(
        trained_triplet_model.layers[0].layers[9].layers[3],
        model_task_item.get_prediction_results_path()
    )


def fine_tune_save_and_predict_siamese(model_task_item: ModelTaskItem, model_inputs):
    # Train Siamese Model.
    trained_siamese_model = train_siamese_model(model_task_item.get_model_name(), *model_inputs)
    trained_siamese_model.save_weights(model_task_item.get_weights_path())

    # Evaluate the trained Siamese model.
    predict_and_save_results(
        trained_siamese_model.layers[0].layers[6].layers[3],
        model_task_item.get_prediction_results_path()
    )


if __name__ == '__main__':
    df, siamese_df, triplet_df = load_datasets(
        Constants.URL_DATASET_PREPROCESSED_TOKENIZED_SDSS,
        Constants.PATH_DATASET_SIAMESE_SDSS,
        Constants.PATH_DATASET_TRIPLET_SDSS
    )
    siamese_df = siamese_df[siamese_df['similarity_score'] == 1]

    sql_tokenizer = load_from_disk_or_call_func(Constants.PATH_TOKENIZER, initialize_sql_base_tokenizer, df)

    tokenized_input_siamese = get_tokenized_siamese_inputs(siamese_df, sql_tokenizer)
    tokenized_inputs_triplets = get_tokenized_triplet_inputs(triplet_df, sql_tokenizer)

    # fine_tune_save_and_predict_triplet(AllModelTaskItems.BERT_TRIPLET.value, tokenized_inputs_triplets)
    # fine_tune_save_and_predict_triplet(AllModelTaskItems.CODEBERT_TRIPLET.value, tokenized_inputs_triplets)

    fine_tune_save_and_predict_siamese(AllModelTaskItems.BERT_SIAMESE.value, tokenized_input_siamese)
    fine_tune_save_and_predict_siamese(AllModelTaskItems.CODEBERT_SIAMESE.value, tokenized_input_siamese)
