'''chit encoder'''
from typing import (
    List,
    Tuple,
)

import tensorflow as tf

from .config import (
    GRU_UNIT_NUM,
)


class ChitEncoder(tf.keras.Model):
    '''encoder'''
    def __init__(
            self,
            embedding: tf.keras.layers.Embedding,
    ) -> None:
        super().__init__()
        self.embedding = embedding

        gru = tf.keras.layers.GRU(
            units=GRU_UNIT_NUM,
            return_sequences=True,
            return_state=True,
            unroll=True,
        )

        self.bi_gru = tf.keras.layers.Bidirectional(
            layer=gru,
            merge_mode='sum',
        )

        self.dropout = tf.keras.layers.Dropout(rate=0.2)

    def call(
            self,
            inputs: List[List[int]],  # shape: [batch_size, max_len]
            training=None,
            # mask=None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.reverse(inputs, [-1])

        outputs = self.embedding(inputs)

        # import pdb; pdb.set_trace()
        if training:
            outputs = self.dropout(outputs)

        [outputs, *hidden_state] = self.bi_gru(outputs)

        hidden_state = tf.concat(hidden_state, axis=1)

        return outputs, hidden_state
