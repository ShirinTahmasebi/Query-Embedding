import pandas as pd

from helper import *
from helper_enum import AllModelTaskItems

if __name__ == '__main__':
    df = pd.read_csv(Constants.URL_DATASET_PREPROCESSED_TOKENIZED)
    sql_based_tokenizer, sql_based_tokenized = sql_based_tokenizer(df)

    fine_tune_model_and_save(
        sql_based_tokenized,
        AllModelTaskItems.BERT_MLM.value
    )
