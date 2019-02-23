'''chat decoder'''
# from typing import (
#     Tuple,
# )

import tensorflow as tf

from .config import (
    DROPOUT_RATE,
    RNN_UNIT_NUM,
)


class ChatDecoder(tf.keras.Model):
    '''decoder'''
    def __init__(
            self,
            voc_size: int,
    ) -> None:
        super().__init__()

        self.voc_size = voc_size

        self.gru = tf.keras.layers.GRU(
            units=RNN_UNIT_NUM * 2,     # for forward & backword gru state
            return_sequences=True,
            return_state=True,
        )

        self.dense = tf.keras.layers.Dense(
            units=self.voc_size,
            activation='softmax',
        )

    def call(
            self,
            inputs: tf.Tensor,
            initial_state: tf.Tensor,
            training=None,
            # mask=None,
    ) -> tf.Tensor:
        '''chat decoder call'''

        # import pdb; pdb.set_trace()

        # print('shape: input:{} initial_state{}'.format(
        #     inputs.shape,
        #     initial_state.shape,
        # ))

        outputs, hidden_state = self.gru(
            inputs=inputs,
            initial_state=[initial_state],
        )

        if training:
            outputs = self.dropout(outputs)

        outputs = self.dense(outputs)

        return outputs, hidden_state
