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

        self.lstm_decoder = tf.keras.layers.LSTM(
            units=LATENT_UNIT_NUM,
            return_sequences=True,
            return_state=True,
        )

        self.dense = tf.keras.layers.Dense(units=voc_size)

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
            teacher_forcing_targets=None,
            training=False,
    ) -> tf.Tensor:
        '''chat decoder call'''
        batch_size = tf.shape(inputs[0])[0]

        batch_go_one_hot = tf.ones([batch_size, 1, 1]) \
            * [tf.one_hot(self.indice_go, self.voc_size)]

        states = inputs

        outputs = tf.zeros([batch_size, 0, self.voc_size])
        output = batch_go_one_hot

        t = 0
        while t < MAX_LEN:
            if training:
                target_indice = teacher_forcing_targets[:, t]
            else:
                target_indice = tf.argmax(output, axis=-1)

            decoder_inputs = tf.expand_dims(
                self.embedding(tf.convert_to_tensor(target_indice)), axis=1)

            output, *states = self.lstm_decoder(
                inputs=decoder_inputs,
                initial_state=states,
            )
            output = self.dense(output)
            outputs = tf.concat([outputs, output], 1)

            t = t + 1

        return outputs
