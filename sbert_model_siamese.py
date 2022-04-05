import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics


# TODO Shirin: Impelement this class!

class SBertModelSiamese(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the two embeddings produced by the
    Siamese Network.

    The siamese loss is defined as:
       L(F, S) = CosineSimilarity(F, S)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SBertModelSiamese, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs, **kwargs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a pair containing cls embedding of the
        # input sentences. (hidden_state of the last encoder, the very first
        # element of each item in the batch. last_hidden_state[:, 0])
        first_embedding, second_embedding = self.siamese_network(data)

        # Computing the Siamese Loss by calculating the cosine similarity of
        # the two embeddings.
        #
        # A part of TF documentation about this loss function:
        # "Note that it is a number between -1 and 1. When it is a negative
        # number between -1 and 0, 0 indicates orthogonality and values closer
        # to -1 indicate greater similarity. The values closer to 1 indicate
        # greater dissimilarity. This makes it usable as a loss function in a
        # setting where you try to maximize the proximity between predictions
        # and targets."
        cosine_similarity = tf.keras.losses.CosineSimilarity(axis=0)
        loss = tf.reduce_sum(cosine_similarity(first_embedding, second_embedding))
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
