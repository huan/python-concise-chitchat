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

        dense = tf.keras.layers.Dense(units=voc_size)
        self.td_dense = tf.keras.layers.TimeDistributed(dense)

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
            teacher_forcing_targets=None,
            training=False,
    ) -> tf.Tensor:
        '''chat decoder call'''
        batch_size = tf.shape(inputs[0])[0]
        # import pdb; pdb.set_trace()

        batch_go_one_hot = tf.ones([batch_size, 1, 1]) \
            * [tf.one_hot(self.indice_go, self.voc_size)]

        states = inputs

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

            output, *states = self.decoder(
                inputs=decoder_inputs,
                initial_state=states,   # (Tensor, Tensor)
                # state_hidden, state_cell
            )
            output = tf.expand_dims(self.dense(output), axis=1)
            outputs = tf.concat([outputs, output], axis=1)

        return outputs
