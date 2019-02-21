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

        self.rnn = tf.keras.layers.GRU(
            units=RNN_UNIT_NUM,
            return_sequences=True,
            return_state=True,
            stateful=True,
        )

    def call(
            self,
            inputs: tf.Tensor,
            initial_state: tf.Tensor,
            training=None,
            # mask=None,
    ) -> tf.Tensor:
        '''chat decoder call'''

        import pdb; pdb.set_trace()

        # self.rnn.reset_states(initial_state)
        xxx = self.rnn(
            inputs=inputs,
            initial_state=[initial_state],
        )

        outputs, hidden_state = xxx

        if training:
            outputs = self.dropout(outputs)

        return outputs, hidden_state
