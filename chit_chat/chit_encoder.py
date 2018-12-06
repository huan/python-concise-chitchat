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

        self.lstm = tf.keras.layers.CuDNNLSTM(
            units=LATENT_UNIT_NUM,
        )

        self.repeat = tf.keras.layers.RepeatVector(MAX_LEN)

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
    ) -> tf.Tensor:
        inputs_embedding = self.embedding(tf.convert_to_tensor(inputs))
        lstm_output = self.lstm(inputs_embedding)
        output = self.repeat(lstm_output)

        return output
