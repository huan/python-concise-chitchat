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

        self.lstm_encoder = tf.keras.layers.CuDNNLSTM(
            units=LATENT_UNIT_NUM,
            return_state=True,
        )

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
    ) -> tf.Tensor:
        inputs_embedding = self.embedding(tf.convert_to_tensor(inputs))
        _, *context = self.lstm_encoder(inputs_embedding)
        return context    # shape: ([latent_unit_num], [latent_unit_num])
