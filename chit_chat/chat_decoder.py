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

        self.initial_state = None

    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
            teacher_forcing_targets=None,
            training=False,
    ) -> tf.Tensor:
        '''chat decoder call'''
        batch_size = tf.shape(inputs)[0]

        batch_go_embedding = tf.ones([batch_size, 1, 1]) \
            * [self.__indice_to_embedding(self.indice_go)]
        batch_go_one_hot = tf.ones([batch_size, 1, 1]) \
            * [tf.one_hot(self.indice_go, self.voc_size)]

        output, *states = self.lstm_decoder(
            batch_go_embedding,
            initial_state=inputs,
        )

        if training:
            teacher_forcing_targets = tf.convert_to_tensor(teacher_forcing_targets)
            teacher_forcing_embeddings = self.embedding(teacher_forcing_targets)

        outputs = batch_go_one_hot

        for t in range(1, MAX_LEN):
            outputs = tf.concat([outputs, self.dense(output)], 1)
            if training:
                target = teacher_forcing_embeddings[:, t, :]
                decoder_input = tf.expand_dims(target, axis=1)
            else:
                indice = tf.argmax(tf.squeeze(output))
                decoder_input = tf.ones([batch_size, 1, 1]) \
                    * [self.__indice_to_embedding(indice)]

            output = self.lstm_decoder(decoder_input, initial_state=states)

        return outputs

    def __indice_to_embedding(self, indice: int) -> tf.Tensor:
        return self.embedding(
            tf.convert_to_tensor(indice))
