from helper import *
from helper_enum import AllModelTaskItems, ModelTaskItem
from helper_prediction import load_bert_sub_model_of_triplet, load_bert_sub_model_of_siamese


def save_prediction_result(path_dict: dict, label_index: str, points_list: list, all_cls_list: list):
    import pickle

    x, y = zip(*points_list)

    with open(path_dict['x'].format('label_' + label_index), 'wb') as file:
        pickle.dump(x, file)

    with open(path_dict['y'].format('label_' + label_index), 'wb') as file:
        pickle.dump(y, file)

    with open(path_dict['cls'].format('label_' + label_index), 'wb') as file:
        pickle.dump(all_cls_list, file)


def predict_and_save_results(fine_tuned_model, path_dict: dict):
    df = pd.read_csv(Constants.PATH_DATASET_TOKENIZED_AND_LABELED)
    sql_tokenizer = load_from_disk_or_call_func(Constants.PATH_TOKENIZER, initialize_sql_base_tokenizer, df)
    sql_tokenized = load_from_disk_or_call_func(
        Constants.PATH_TOKENIZED_DF, sql_based_tokenizer, df.full_query, sql_tokenizer
    )[1]

    for label_index in range(12):
        sql_based_tokenized = {
            'input_ids': tensor_torch_to_tf(sql_tokenized['input_ids'][df['labels'] == label_index]),
            'attention_mask': tensor_torch_to_tf(sql_tokenized['attention_mask'][df['labels'] == label_index]),
            'token_type_ids': tensor_torch_to_tf(sql_tokenized['token_type_ids'][df['labels'] == label_index])
        }

        batch_size = 20
        data_count = len(sql_based_tokenized['input_ids'])
        points_list = []
        all_cls_list = []
        continue_loop = True

        for i in range(0, data_count, batch_size):
            lower_bound = i
            upper_bound = lower_bound + batch_size  # Supposed to be exclusive
            if (data_count - lower_bound) % batch_size < 5:
                upper_bound += batch_size
                continue_loop = False

            if not lower_bound < data_count:
                break

            sql_based_tokenized_temp = {
                'input_ids': sql_based_tokenized['input_ids'][lower_bound:upper_bound].numpy(),
                'attention_mask': sql_based_tokenized['attention_mask'][lower_bound:upper_bound].numpy(),
                'token_type_ids': sql_based_tokenized['token_type_ids'][lower_bound:upper_bound].numpy()
            }

            cls_list = fine_tuned_model(sql_based_tokenized_temp).last_hidden_state[:, 0, :]

            # from sklearn.manifold import TSNE

            # tsne = TSNE(n_components=2, random_state=0)
            # cls_list_2d = tsne.fit_transform(cls_list)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            cls_list_2d = pca.fit_transform(cls_list)

            points_list = [*points_list, *cls_list_2d]
            all_cls_list = [*all_cls_list, *cls_list]

            save_prediction_result(path_dict, str(label_index), points_list, all_cls_list)

            if not continue_loop:
                break


if __name__ == '__main__':
    task: ModelTaskItem = AllModelTaskItems.CODEBERT_TRIPLET.value
    code_bert_fine_tuned_with_triplet_model = load_bert_sub_model_of_triplet(task)
    predict_and_save_results(code_bert_fine_tuned_with_triplet_model, task.get_prediction_results_path())

    task: ModelTaskItem = AllModelTaskItems.BERT_TRIPLET.value
    bert_fine_tuned_with_triplet_model = load_bert_sub_model_of_triplet(task)
    predict_and_save_results(bert_fine_tuned_with_triplet_model, task.get_prediction_results_path())

    task: ModelTaskItem = AllModelTaskItems.CODEBERT_SIAMESE.value
    code_bert_fine_tuned_with_siamese_model = load_bert_sub_model_of_siamese(task)
    predict_and_save_results(code_bert_fine_tuned_with_siamese_model, task.get_prediction_results_path())

    task: ModelTaskItem = AllModelTaskItems.BERT_SIAMESE.value
    bert_fine_tuned_with_siamese_model = load_bert_sub_model_of_siamese(task)
    predict_and_save_results(bert_fine_tuned_with_siamese_model, task.get_prediction_results_path())
