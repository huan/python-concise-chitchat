'''chat decoder'''
from typing import (
    Tuple,
)

import tensorflow as tf

from .config import (
    LATENT_UNIT_NUM,
    MAX_LEN,
)


class ChatDecoder(tf.keras.Model):
    '''decoder'''
    def __init__(
            self,
            embedding: tf.keras.layers.Embedding,
            voc_size: int,
            indice_go: int,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.indice_go = indice_go
        self.voc_size = voc_size

        lstm = tf.keras.layers.CuDNNLSTM(
            units=LATENT_UNIT_NUM,
            return_sequences=True,
        )
        self.bi_lstm = tf.keras.layers.Bidirectional(lstm)

        self.bidirectional_lstm = tf.keras.layers.Bidirectional(lstm)
        self.batch_normalization = tf.keras.layers.BatchNormalization()

        self.time_distributed = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=voc_size
            )
        )

    def call(
            self,
            inputs: tf.Tensor,
    ) -> tf.Tensor:
        '''chat decoder call'''

        outputs = self.bidirectional_lstm(inputs)
        outputs =

        outputs = tf.zeros([batch_size, 0, self.voc_size])
        output = batch_go_one_hot

        for t in range(0, MAX_LEN):
            # import pdb; pdb.set_trace()
            if training and teacher_forcing_targets is not None:
                target_indice = tf.expand_dims(
                    teacher_forcing_targets[:, t], axis=-1)
            else:
                target_indice = tf.argmax(output, axis=-1)

            decoder_inputs = self.embedding(target_indice)

            output, *states = self.bidirectional_lstm(
                inputs=decoder_inputs,
                initial_state=states,   # (Tensor, Tensor)
                # state_hidden, state_cell
            )
            output = tf.expand_dims(self.dense(output), axis=1)
            outputs = tf.concat([outputs, output], axis=1)

        return outputs
