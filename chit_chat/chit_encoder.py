'''chit encoder'''
from typing import (
    List,
    Tuple,
)

import tensorflow as tf

from .config import (
    DROPOUT_RATE,
    RNN_UNIT_NUM,
)


class ChitEncoder(tf.keras.Model):
    '''encoder'''
    def __init__(
            self,
    ) -> None:
        super().__init__()
        gru = tf.keras.layers.GRU(
            units=RNN_UNIT_NUM,
            return_sequences=True,
            return_state=True,
            unroll=True,
        )

        self.bi_gru = tf.keras.layers.Bidirectional(
            layer=gru,
            merge_mode='concat',
        )

        self.dropout = tf.keras.layers.Dropout(rate=DROPOUT_RATE)

    def call(
            self,
            inputs: tf.Tensor,  # shape: [batch_size, max_len, embedding_dim]
            training=None,
            # mask=None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # import pdb; pdb.set_trace()

        [outputs, hidden_state_forward, hidden_state_backword] = self.bi_gru(inputs)

        state = tf.concat([hidden_state_forward, hidden_state_backword], axis=1)
        return outputs, state
