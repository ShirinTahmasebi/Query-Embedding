import pandas as pd

from constants import Constants


def plot_all_labels():
    import pickle

    x = pickle.load(open(Constants.PATH_PREDICTED_TRIPLET_BERT_2D_X, 'rb'))
    y = pickle.load(open(Constants.PATH_PREDICTED_TRIPLET_BERT_2D_Y, 'rb'))
    labels = pickle.load(open(Constants.PATH_PREDICTED_TRIPLET_BERT_LABELS, 'rb'))

    import matplotlib.pyplot as plt
    import pandas as pd

    color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
                 8: 'plum', 9: 'yellowgreen', 10: 'darkorchid', 11: 'darkgreen'}

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for idx, cl in enumerate(range(10)):
        axes[int(cl / 3), cl % 3].scatter(
            x=pd.Series(x)[(pd.Series(labels) == cl).tolist()].tolist(),
            y=pd.Series(y)[(pd.Series(labels) == cl).tolist()].tolist(),
            color=color_map[cl]
        )
    plt.show()


def plot_each_label(path_dict: dict):
    import matplotlib.pyplot as plt
    import pickle

    color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
                 8: 'plum', 9: 'yellowgreen', 10: 'darkorchid', 11: 'darkgreen'}

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for idx, label_index in enumerate(range(12)):
        x = pickle.load(open(path_dict['x'].format('label_' + str(label_index)), 'rb'))
        y = pickle.load(open(path_dict['y'].format('label_' + str(label_index)), 'rb'))
        axes[int(label_index / 3), label_index % 3].scatter(
            x=x,
            y=y,
            color=color_map[label_index]
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
            similarity_score += np.dot(item_one, item_two) / (norm(item_one) * norm(item_two))

    return similarity_score / (len(list_one) * len(list_two))


def calculate_cosine_similarity(path_dict: dict):
    import pickle
    cls_path = path_dict['cls']
    all_cls_dict = {}
    for label_index in range(12):
        cls = pickle.load(open(cls_path.format('label_' + str(label_index)), 'rb'))
        all_cls_dict['label_' + str(label_index)] = cls

    cosine_similarity_dict = {}

    for i in range(12):
        for j in range(i, 12):
            src_label = 'label_' + str(i)
            dest_label = 'label_' + str(j)
            cosine_similarity_dict[(src_label, dest_label)] = \
                calculate_cosine_similarity_by_label(src_label, dest_label, all_cls_dict)

    from sklearn import preprocessing
    import numpy as np
    min_max_scaler = preprocessing.MinMaxScaler()
    score_scaled = min_max_scaler.fit_transform(np.array(list(cosine_similarity_dict.values())).reshape(-1,1))
    df = pd.DataFrame(columns=['labels', 'score'])
    df['score'] = np.squeeze(score_scaled)
    df['labels'] = list(cosine_similarity_dict.keys())
    return df


if __name__ == '__main__':
    plot_each_label(Constants.TRIPLET_BERT_RESULT_PATH_DIC)
    calculate_cosine_similarity(Constants.TRIPLET_BERT_RESULT_PATH_DIC)
