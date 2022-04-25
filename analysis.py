import pandas as pd

from constants import Constants
from dataset_models import Dataset
from dataset_models_bombay import DatasetBombay


def plot_all_labels(path_dict: dict, dataset: Dataset):
    # TODO Shirin: Extract the duplicated part!
    df = pd.read_csv(dataset.get_path_labeled_tokenized_dataset())
    for key in path_dict.keys():
        path_dict[key] = path_dict[key].format(database=dataset.get_name(), label='{label}')

    import matplotlib.pyplot as plt
    import pickle
    import numpy as np

    labels_unique_list = np.unique(df[df['labels'] != -1]['labels'])

    for label_index in labels_unique_list:
        x = pickle.load(open(path_dict['x'].format(label='label_' + str(label_index)), 'rb'))
        y = pickle.load(open(path_dict['y'].format(label='label_' + str(label_index)), 'rb'))
        plt.scatter(x, y, alpha=0.3, label=label_index)
    plt.show()


def plot_each_label(path_dict: dict, dataset: Dataset):
    df = pd.read_csv(dataset.get_path_labeled_tokenized_dataset())
    for key in path_dict.keys():
        path_dict[key] = path_dict[key].format(database=dataset.get_name(), label='{label}')

    import matplotlib.pyplot as plt
    import pickle
    import math
    import numpy as np

    labels_unique_list = np.unique(df[df['labels'] != -1]['labels'])

    color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
                 8: 'plum', 9: 'yellowgreen', 10: 'darkorchid', 11: 'darkgreen'}

    fig, axes = plt.subplots(nrows=math.ceil(len(labels_unique_list) / 3), ncols=3, figsize=(10, 10))
    for label_index in labels_unique_list:
        x = pickle.load(open(path_dict['x'].format(label='label_' + str(label_index)), 'rb'))
        y = pickle.load(open(path_dict['y'].format(label='label_' + str(label_index)), 'rb'))
        axes[int(label_index / 3), label_index % 3].scatter(
            x=x,
            y=y,
            color=color_map[label_index % len(color_map)],
            alpha=0.3,
            label=label_index
        )
    plt.show()


def calculate_cosine_similarity_by_label(label_one, label_two, all_cls_dict):
    import numpy as np
    from numpy.linalg import norm

    list_one = all_cls_dict[label_one]
    list_two = all_cls_dict[label_two]

    similarity_score = 0
    for item_one in list_one:
        for item_two in list_two:
            similarity_score += np.dot(item_one.detach().numpy(), item_two.detach().numpy()) / \
                                (norm(item_one.detach().numpy()) * norm(item_two.detach().numpy()))

    return similarity_score / (len(list_one) * len(list_two))


def calculate_cosine_similarity(path_dict: dict):
    # TODO Tahmasebi: Remove the hard-coded label numbers.
    import pickle
    cls_path = path_dict['cls']
    all_cls_dict = {}
    for label_index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        cls = pickle.load(open(cls_path.format(label='label_' + str(label_index)), 'rb'))
        all_cls_dict['label_' + str(label_index)] = cls

    cosine_similarity_dict = {}

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        for j in range(i, 14):
            src_label = 'label_' + str(i)
            dest_label = 'label_' + str(j)
            cosine_similarity_dict[(src_label, dest_label)] = \
                calculate_cosine_similarity_by_label(src_label, dest_label, all_cls_dict)

    from sklearn import preprocessing
    import numpy as np
    min_max_scaler = preprocessing.MinMaxScaler()
    score_scaled = min_max_scaler.fit_transform(np.array(list(cosine_similarity_dict.values())).reshape(-1, 1))
    df = pd.DataFrame(columns=['labels', 'score'])
    df['score'] = np.squeeze(score_scaled)
    df['labels'] = list(cosine_similarity_dict.keys())
    return df


if __name__ == '__main__':
    plot_all_labels(Constants.TRIPLET_BERT_RESULT_PATH_DIC_TEMP, DatasetBombay())
    plot_each_label(Constants.TRIPLET_BERT_RESULT_PATH_DIC_TEMP, DatasetBombay())
    df_cosine_similarity_triplet_code_bert = calculate_cosine_similarity(Constants.TRIPLET_BERT_RESULT_PATH_DIC_TEMP)

    # plot_all_labels(Constants.TRIPLET_BERT_RESULT_PATH_DIC)
    # plot_all_labels(Constants.SIAMESE_BERT_RESULT_PATH_DIC)
    # plot_all_labels(Constants.SIAMESE_CODE_BERT_RESULT_PATH_DIC)

    # plot_each_label(Constants.TRIPLET_BERT_RESULT_PATH_DIC)
    # plot_each_label(Constants.SIAMESE_BERT_RESULT_PATH_DIC)
    # plot_each_label(Constants.SIAMESE_CODE_BERT_RESULT_PATH_DIC)

    # df_cosine_similarity_triplet_bert = calculate_cosine_similarity(Constants.TRIPLET_BERT_RESULT_PATH_DIC)
    # df_cosine_similarity_siamese_bert = calculate_cosine_similarity(Constants.SIAMESE_BERT_RESULT_PATH_DIC)
    # df_cosine_similarity_siamese_code_bert = calculate_cosine_similarity(Constants.SIAMESE_CODE_BERT_RESULT_PATH_DIC)

    pass
