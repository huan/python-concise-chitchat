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
            return_state=True,
        )

        # self.encoder = lstm
        self.encoder = tf.keras.layers.Bidirectional(lstm)

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
    ) -> tf.Tensor:
        inputs_embedding = self.embedding(tf.convert_to_tensor(inputs))
        _, *bi_context = self.encoder(inputs_embedding)

        # import pdb; pdb.set_trace()

        forward_hidden, forward_cell, reverse_hidden, reverse_cell = bi_context
        context = (
            tf.concat([forward_hidden, reverse_hidden], axis=1),
            tf.concat([forward_cell, reverse_cell], axis=1)
        )

        return context    # shape: ([latent_unit_num], [latent_unit_num])
