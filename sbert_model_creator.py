from constants import Constants
from sbert_distance_layer import DistanceLayerTriplet

from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
from transformers import TFAutoModel


def initialize_embedding_model():
    bert_model = TFAutoModel.from_pretrained(Constants.BERT_MODEL_NAME, output_hidden_states=True)

    input_layer_input_ids = tf.keras.layers.Input(shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    input_layer_attention_mask = tf.keras.layers.Input(shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    input_layer_token_type_ids = tf.keras.layers.Input(shape=(Constants.MAX_LENGTH,), dtype=tf.int64)

    bert_layer = bert_model(
        input_ids=input_layer_input_ids,
        attention_mask=input_layer_attention_mask,
        token_type_ids=input_layer_token_type_ids
    )

    bert_layer = bert_layer.last_hidden_state[:, 0, :]

    embedding_model = tf.keras.Model(
        inputs=[input_layer_input_ids, input_layer_attention_mask, input_layer_token_type_ids],
        outputs=bert_layer,
        name='embedding_model'
    )

    return embedding_model


def create_sbert_model_siamese():
    embedding_model = initialize_embedding_model()

    first_input_ids = layers.Input(name="1st_input_ids", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    first_attention_mask = layers.Input(name="1st_attention", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    first_token_type_ids = layers.Input(name="1st_token_types", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)

    second_input_ids = layers.Input(name="2nd_input_ids", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    second_attention_mask = layers.Input(name="2nd_attention", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    second_token_type_ids = layers.Input(name="2nd_token_types", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)

    first_cls = embedding_model([first_input_ids, first_attention_mask, first_token_type_ids])
    second_cls = embedding_model([second_input_ids, second_attention_mask, second_token_type_ids])

    siamese_network = Model(
        inputs=[
            first_input_ids, first_attention_mask, first_token_type_ids,
            second_input_ids, second_attention_mask, second_token_type_ids
        ], outputs=(first_cls, second_cls)
    )

    return siamese_network


def create_sbert_model_triplet():
    embedding_model = initialize_embedding_model()

    anchor_input_input_ids = layers.Input(name="anchor_input_ids", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    anchor_input_attention_mask = layers.Input(name="anchor_attention", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    anchor_input_token_type_ids = layers.Input(name="anchor_token_types", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)

    positive_input_input_ids = layers.Input(name="pos_input_ids", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    positive_input_attention_mask = layers.Input(name="pos_attention", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    positive_input_token_type_ids = layers.Input(name="pos_token_types", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)

    negative_input_input_ids = layers.Input(name="neg_input_ids", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    negative_input_attention_mask = layers.Input(name="neg_attention", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)
    negative_input_token_type_ids = layers.Input(name="neg_token_types", shape=(Constants.MAX_LENGTH,), dtype=tf.int64)

    distances = DistanceLayerTriplet()(
        embedding_model([anchor_input_input_ids, anchor_input_attention_mask, anchor_input_token_type_ids]),
        embedding_model([positive_input_input_ids, positive_input_attention_mask, positive_input_token_type_ids]),
        embedding_model([negative_input_input_ids, negative_input_attention_mask, negative_input_token_type_ids]),
    )

    triplet_network = Model(
        inputs=[
            anchor_input_input_ids, anchor_input_attention_mask, anchor_input_token_type_ids,
            positive_input_input_ids, positive_input_attention_mask, positive_input_token_type_ids,
            negative_input_input_ids, negative_input_attention_mask, negative_input_token_type_ids
        ], outputs=distances
    )

    return triplet_network
