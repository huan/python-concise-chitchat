'''chit encoder'''
from typing import (
    List,
)

import tensorflow as tf

from .config import (
    LATENT_UNIT_NUM,
    MAX_LEN,
)


class ChitEncoder(tf.keras.Model):
    '''encoder'''
    def __init__(
            self,
            embedding: tf.keras.layers.Embedding,
    ) -> None:
        super().__init__()
        self.embedding = embedding

        lstm = tf.keras.layers.CuDNNLSTM(
            units=LATENT_UNIT_NUM,
        )

        # self.encoder = lstm
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(lstm)
        self.batch_normalzation = tf.keras.layers.BatchNormalization()

        self.repeat_vector = tf.keras.layers.RepeatVector(MAX_LEN)

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
    ) -> tf.Tensor:
        inputs_embedding = self.embedding(tf.convert_to_tensor(inputs))

        output = self.bidirectional_lstm(inputs_embedding)
        output = self.batch_normalzation(output)

        outputs = self.repeat_vector(output)

        return outputs    # shape: ([latent_unit_num], [latent_unit_num])
