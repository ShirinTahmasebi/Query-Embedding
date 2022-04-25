from transformers import BertModel

from dataset_models import Dataset
from dataset_models_bombay import DatasetBombay
from helper import *
from helper_enum import AllModelTaskItems, ModelTaskItem
from helper_prediction import load_bert_sub_model_of_triplet, load_bert_sub_model_of_siamese


def save_prediction_result(path_dict: dict, label_index: str, points_list: list, all_cls_list: list):
    from pathlib import Path
    path_str_x = path_dict['x'].format(label='label_' + label_index)
    path_str_y = path_dict['y'].format(label='label_' + label_index)
    path_str_cls = path_dict['cls'].format(label='label_' + label_index)

    path_x = Path(path_str_x)
    path_y = Path(path_str_y)
    path_cls = Path(path_str_cls)

    Path(path_x.parent).mkdir(parents=True, exist_ok=True)

    Path(path_y.parent).mkdir(parents=True, exist_ok=True)

    Path(path_cls.parent).mkdir(parents=True, exist_ok=True)

    import pickle

    x, y = zip(*points_list)

    with open(path_str_x, 'wb') as file:
        pickle.dump(x, file)

    with open(path_str_y, 'wb') as file:
        pickle.dump(y, file)

    with open(path_str_cls, 'wb') as file:
        pickle.dump(all_cls_list, file)


def predict_and_save_results(fine_tuned_model, path_dict: dict, dataset: Dataset):
    for key in path_dict.keys():
        path_dict[key] = path_dict[key].format(database=dataset.get_name(), label='{label}')

    df = pd.read_csv(dataset.get_path_labeled_tokenized_dataset())
    sql_tokenizer = load_from_disk_or_call_func(dataset.get_path_tokenizer(), initialize_sql_base_tokenizer, df)
    sql_tokenized = load_from_disk_or_call_func(
        dataset.get_tokenized_path(), sql_based_tokenizer, df['full_query'], sql_tokenizer
    )[1]  # TODO Shirin: Delete tokenized df files. Remove this index. Rerun!

    import numpy as np
    labels_unique_list = np.unique(df[df['labels'] != -1]['labels'])
    for label_index in labels_unique_list:
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

            # TODO Shirin: Here, I have converted the inputs from np.arr to torch.tensor.
            # TODO Shirin: Check if it is OK for other models.
            # outputs = model(
            #     input_ids=torch.from_numpy(input_ids),
            #     attention_mask=torch.from_numpy(attention_mask),
            #     token_type_ids=torch.from_numpy(token_type_ids)
            # )
            cls_list = fine_tuned_model(
                input_ids=torch.from_numpy(sql_based_tokenized_temp['input_ids']),
                attention_mask=torch.from_numpy(sql_based_tokenized_temp['attention_mask']),
                token_type_ids=torch.from_numpy(sql_based_tokenized_temp['token_type_ids'])
            ).last_hidden_state[:, 0, :]

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            cls_list_2d = pca.fit_transform(cls_list.detach().numpy())

            points_list = [*points_list, *cls_list_2d]
            all_cls_list = [*all_cls_list, *cls_list]

            if not continue_loop:
                break

        save_prediction_result(path_dict, str(label_index), points_list, all_cls_list)




if __name__ == '__main__':
    dataset = DatasetBombay()

    task: ModelTaskItem = AllModelTaskItems.BERT_TRIPLET_TEMP.value
    # code_bert_fine_tuned_with_triplet_model = load_bert_sub_model_of_triplet(task, dataset)
    model = BertModel.from_pretrained(Constants.BERT_MODEL_NAME_BASE_UNCASED, output_hidden_states=True)
    model.load_state_dict(torch.load(Constants.PATH_FINE_TUNED_MODEL_WEIGHTS_TEMP_BERT))
    model.eval()

    predict_and_save_results(
        model,
        task.get_prediction_results_path(),
        dataset
    )
