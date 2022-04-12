from fine_tuning_data_preparation import DataPreparation
from helper import *
from helper_enum import AllModelTaskItems, ModelTaskItem


def fine_tune_model_and_save(tokenized, model_task_item: ModelTaskItem):
    model = fetch_model(model_task_item.get_model())
    fine_tuning_task = model_task_item.get_task()
    model_name = model_task_item.get_prefix_model_name()

    data_processor = DataPreparation.init_processor(fine_tuning_task)
    train_dataset, eval_dataset = create_torch_dataset(tokenized, data_processor)

    training_loop(
        model=model,
        train_dataset=train_dataset
    )

    import pickle
    pickle.dump(model, open(model_name, 'wb'))


if __name__ == '__main__':
    df = pd.read_csv(Constants.URL_DATASET_PREPROCESSED_TOKENIZED_SDSS)
    sql_tokenizer = initialize_sql_base_tokenizer(df)
    sql_based_tokenizer, sql_based_tokenized = sql_based_tokenizer(df.full_query, sql_tokenizer)

    fine_tune_model_and_save(
        sql_based_tokenized,
        AllModelTaskItems.BERT_MLM.value
    )
