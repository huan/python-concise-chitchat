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

        self.lstm_encoder = tf.keras.layers.LSTM(
            units=LATENT_UNIT_NUM,
            return_state=True,
        )

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
            training=None,
            mask=None,
    ) -> tf.Tensor:
        inputs_embedding = self.embedding(tf.convert_to_tensor(inputs))
        _, *state = self.lstm_encoder(inputs_embedding)
        return state    # shape: ([latent_unit_num], [latent_unit_num])
