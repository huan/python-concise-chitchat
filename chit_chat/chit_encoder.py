'''chit encoder'''
from typing import (
    List,
)

import tensorflow as tf

from .config import (
    LATENT_UNIT_NUM,
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

        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
            training=None,
            mask=None,
    ) -> tf.Tensor:
        outputs = tf.convert_to_tensor(inputs)
        outputs = self.embedding(outputs)
        outputs = self.bidirectional_lstm(outputs)

        if training:
            outputs = self.dropout(outputs)

        return outputs    # shape: ([latent_unit_num], [latent_unit_num])
