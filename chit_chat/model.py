'''doc'''
import tensorflow as tf
import numpy as np
from typing import (
    List,
    # Tuple,
)

from .vocabulary import Vocabulary
from .config import (
    DONE,
    GO,
    MAX_LEN,
)

EMBEDDING_DIM = 100
LATENT_UNIT_NUM = 300


class ChitEncoder(tf.keras.Model):
    def __init__(
            self,
    ) -> None:
        super().__init__()

        self.lstm_encoder = tf.keras.layers.LSTM(
            units=LATENT_UNIT_NUM,
            return_state=True,
        )

    def call(
            self,
            inputs: tf.Tensor,  # shape: [batch_size, max_len, embedding_dim]
            training=None,
            mask=None,
    ) -> tf.Tensor:
        _, *state = self.lstm_encoder(inputs)
        return state    # shape: ([latent_unit_num], [latent_unit_num])


class ChatDecoder(tf.keras.Model):
    def __init__(
            self,
            voc_size: int,
    ) -> None:
        super().__init__()

        self.lstm_decoder = tf.keras.layers.LSTM(
            units=LATENT_UNIT_NUM,
            return_sequences=True,
            return_state=True,
        )

        self.dense = tf.keras.layers.Dense(
            units=voc_size,
        )

        self.time_distributed_dense = tf.keras.layers.TimeDistributed(
            self.dense
        )

        self.initial_state = None

    def set_state(self, state=None):
        # import pdb; pdb.set_trace()
        self.initial_state = state

    def call(
            self,
            inputs: tf.Tensor,  # shape: [batch_size, None, embedding_dim]
            training=False,
            mask=None,
    ) -> tf.Tensor:
        '''chat decoder call'''

        batch_size = tf.shape(inputs)[0]
        max_len = tf.shape(inputs)[0]

        # outputs = tf.zeros(shape=(
        #     batch_size,         # batch_size
        #     max_len,            # max time step
        #     LATENT_UNIT_NUM,    # dimention of hidden state
        # ))

        import pdb; pdb.set_trace()
        outputs, *states = self.lstm_decoder(inputs, initial_state=self.initial_state)
        self.initial_state = states

        outputs = self.time_distributed_dense(outputs)
        return outputs


class ChitChat(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary: Vocabulary,
    ) -> None:
        super().__init__()

        self.word_index = vocabulary.tokenizer.word_index
        self.voc_size = vocabulary.size

        # [batch_size, max_len] -> [batch_size, max_len, voc_size]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.voc_size,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,
        )

        self.encoder = ChitEncoder()
        # shape: [batch_size, state]

        self.decoder = ChatDecoder(self.voc_size)
        # shape: [batch_size, max_len, voc_size]

    def call(
            self,
            inputs: tf.Tensor,  # shape: [batch_size, max_len]
            teacher_forcing_targets: tf.Tensor=None,  # shape: [batch_size, max_len]
            training=None,
            mask=None,
    ) -> tf.Tensor:     # shape: [batch_size, max_len, embedding_dim]
        '''call'''
        batch_size = tf.shape(inputs)[0]

        inputs_embedding = self.embedding(tf.convert_to_tensor(inputs))
        state = self.encoder(inputs_embedding)

        self.decoder.set_state(state)

        # numpy.ndarray or tf.Tensor ???
        # outputs = tf.Variable(
        #     tf.zeros(shape=(
        #         batch_size,             # batch_size
        #         MAX_LEN,        # max time step
        #         LATENT_UNIT_NUM,          # dimention of hidden state
        #     )),
        #     dtype=tf.float32,
        # )
        if training:
            teacher_forcing_targets = tf.convert_to_tensor(teacher_forcing_targets)
            teacher_forcing_embeddings = self.embedding(teacher_forcing_targets)

        # outputs[:, 0, :].assign([self.__go_embedding()] * batch_size)
        batch_go_embedding = tf.ones([batch_size, 1, 1]) * [self.__go_embedding()]
        outputs = batch_go_embedding
        output = self.decoder(batch_go_embedding)

        for t in range(1, MAX_LEN):
            # outputs[:, t, :] = output
            outputs = tf.concat([outputs, output], 1)
            if training:
                target = teacher_forcing_embeddings[:, t, :]
                # target = tf.expand_dims(target, axis=1)
                output = target

            output = self.decoder(output)

        return outputs

    def predict(self, inputs: List[int], temperature=1.) -> List[int]:
        '''doc'''

        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.expand_dims(inputs, axis=0)     # shape: [1, len(inputs)]

        embeddings = self.embedding(inputs)
        # shape: [1, len(inputs), embedding_dim]

        state = self.encoder(embeddings)
        self.decoder.set_states(state)

        outputs = np.zeros((MAX_LEN,))
        outputs[0] = self.word_index[GO]

        for t in range(1, MAX_LEN):
            output = self.decoder(outputs[t - 1])

            # align the embedding value
            indice = self.__logit_to_indice(output, temperature=temperature)

            outputs[t] = indice

            if indice == self.word_index[DONE]:
                break

        return outputs

    def __go_embedding(self) -> tf.Tensor:
        return self.embedding(
            tf.convert_to_tensor(self.word_index[GO]))

    def __logit_to_indice(
            self,
            inputs,
            temperature=1.,
    ) -> int:
        '''
        [vocabulary_size]
        convert one hot encoding to indice with temperature
        '''
        inputs = tf.squeeze(inputs)
        prob = tf.nn.softmax(inputs / temperature).numpy()
        indice = np.random.choice(self.voc_size, p=prob)
        return indice

    def __indice_to_embedding(self, indice: int) -> tf.Tensor:
        tensor = tf.convert_to_tensor([[indice]])
        return self.embedding(tensor)
