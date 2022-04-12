import pandas as pd
import tensorflow as tf
import torch

from constants import Constants
from helper import load_from_disk_or_call_func, sql_based_tokenizer
from helper_enum import ModelTaskItem
from sbert_model_creator import create_sbert_model_triplet, create_sbert_model_siamese
from sbert_model_siamese import SBertModelSiamese
from sbert_model_triplet import SBertModelTriplet


def load_bert_sub_model_of_siamese(model_task_item: ModelTaskItem):
    bert_model_name = model_task_item.get_model_name()
    weights_path = model_task_item.get_weights_path()

    siamese_model_loaded = SBertModelSiamese(create_sbert_model_siamese(bert_model_name))
    siamese_model_loaded.load_weights(weights_path)
    return siamese_model_loaded.layers[0].layers[6].layers[3]


def load_bert_sub_model_of_triplet(model_task_item: ModelTaskItem):
    bert_model_name = model_task_item.get_model_name()
    weights_path = model_task_item.get_weights_path()

    triplet_model_loaded = SBertModelTriplet(create_sbert_model_triplet(bert_model_name))
    triplet_model_loaded.load_weights(weights_path)
    return triplet_model_loaded.layers[0].layers[9].layers[3]


def load_datasets(*dataset_names):
    loaded_datasets = []
    for dataset_name in dataset_names:
        loaded_datasets.append(pd.read_csv(dataset_name))
    return loaded_datasets


def get_tokenized_siamese_inputs(siamese_df: pd.DataFrame, sql_tokenizer):
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

    return tokenized_input_siamese_one, tokenized_input_siamese_two


def get_tokenized_triplet_inputs(triplet_df: pd.DataFrame, sql_tokenizer):
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

    return tokenized_input_triplet_anchor, tokenized_input_triplet_positive, tokenized_input_triplet_negative


def train_siamese_model(
        bert_model_name,
        tokenized_input_siamese_one,
        tokenized_input_siamese_two
):
    from tensorflow.keras import optimizers

    siamese_network = SBertModelSiamese(create_sbert_model_siamese(bert_model_name))
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

    return siamese_network


def train_triplet_model(
        bert_model_name,
        tokenized_input_triplet_anchor,
        tokenized_input_triplet_positive,
        tokenized_input_triplet_negative
):
    from tensorflow.keras import optimizers

    triplet_network = SBertModelTriplet(create_sbert_model_triplet(bert_model_name))
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
        steps_per_epoch=50,
        validation_steps=5
    )

    return triplet_network


def predict(model, loader):
    predictions = []
    labels = []
    for batch in loader:
        dup_batch = batch.copy()
        batch.pop('session_id')
        outputs = model(**batch)
        predictions.append(torch.argmax(outputs.logits, dim=-1))
        labels.append(dup_batch['session_id'])
    predictions_list = [item for sublist in predictions for item in sublist.numpy()]
    label_list = [item for sublist in labels for item in sublist.numpy()]
    return predictions_list, label_list


def plot_results(predictions_list, label_list):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np

    standard = StandardScaler()
    x_std = standard.fit_transform(predictions_list)
    plt.figure()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(label_list)

    tsne = TSNE(n_components=2, random_state=0)
    x_test_2d = tsne.fit_transform(x_std)

    markers = ('s', 'd', 'o', '^', 'v', '8', 's', 'p', "_", '2')
    color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
                 8: 'plum', 9: 'yellowgreen', 10: 'darkorchid', 11: 'darkgreen'}
    # colors = cm.rainbow(np.linspace(0, 1, len(label_list)))

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=x_test_2d[y == cl, 0], y=x_test_2d[y == cl, 1], c=color_map[idx % len(markers)],
                   marker=markers[idx % len(markers)])  # , label=label_list)

    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(label_list, loc='upper left')  # label_list
    plt.title('t-SNE visualization of test data')
    plt.xlim([min(x_test_2d[:, 0]), max(x_test_2d[:, 0])])
    plt.ylim([min(x_test_2d[:, 1]), max(x_test_2d[:, 1])])
    plt.show()
