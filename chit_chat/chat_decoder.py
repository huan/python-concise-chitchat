'''chat decoder'''
# from typing import (
#     Tuple,
# )

import tensorflow as tf

from .config import (
    LATENT_UNIT_NUM,
)


class ChatDecoder(tf.keras.Model):
    '''decoder'''
    def __init__(
            self,
            embedding: tf.keras.layers.Embedding,
            voc_size: int,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.voc_size = voc_size

        # lstm = tf.keras.layers.GRU(
        lstm = tf.keras.layers.LSTM(
            units=LATENT_UNIT_NUM,
            return_sequences=True,
        )

        self.bi_lstm = tf.keras.layers.Bidirectional(lstm)

        self.time_distributed = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=voc_size
            )
        )
        # self.dropout = tf.keras.layers.Dropout(rate=0.2)

    def call(
            self,
            inputs: tf.Tensor,
            training=None,
            # mask=None,
    ) -> tf.Tensor:
        '''chat decoder call'''

        outputs = inputs

        outputs = self.bi_lstm(outputs)

        # if training:
        #     outputs = self.dropout(outputs)

        outputs = self.time_distributed(outputs)

        return outputs
