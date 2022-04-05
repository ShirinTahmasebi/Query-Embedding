from tensorflow.keras import layers
import tensorflow as tf


# TODO Shirin: Probably useless and can be removed!
class DistanceLayerSiamese(layers.Layer):
    """
    This layer is responsible for computing the distance between the two
    embeddings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, query_one, query_two):
        # TODO Shirin: Maybe it would be better to replace it with cosine similarity.
        distance = tf.reduce_sum(tf.square(query_one - query_two))
        return distance


class DistanceLayerTriplet(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance
