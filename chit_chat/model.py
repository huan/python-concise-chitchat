'''doc'''
import tensorflow as tf
import numpy as np
from typing import (
    List,
)

from .vocabulary import Vocabulary
from .config import (
    DONE,
    GO,
    MAX_LENGTH,
)

EMBEDDING_DIMENTION = 100
LSTM_UNIT_NUM = 300


class ChitChat(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary: Vocabulary,
    ) -> None:
        super().__init__()

        self.word_index = vocabulary.tokenizer.word_index
        self.vocabulary_size = vocabulary.size

        # [batch_size, max_length] -> [batch_size, max_length, vocabulary_size]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=EMBEDDING_DIMENTION,
            mask_zero=True,
        )
        self.lstm_encoder = tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            return_state=True,
          )
        )
        self.lstm_decoder = tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            return_state=True,
            return_sequences=True,
          )
        )

        self.dense = tf.keras.layers.Dense(
            units=self.vocabulary_size,
        )

        self.time_distributed_dense = tf.keras.layers.TimeDistributed(
            self.dense
        )

    def call(
            self,
            inputs: tf.Tensor,
            teacher_forcing_inputs: tf.Tensor=None,
            training=None,
    ) -> tf.Tensor:
        '''call'''

        if not training:
            return self.__call_for_normal(inputs)

        if teacher_forcing_inputs is None:
            raise ValueError('decoder_inputs not set when training')

        return self.__call_for_training(inputs, teacher_forcing_inputs)

    def __call_for_training(
            self,
            queries: tf.Tensor,
            teacher_forcing_responses: tf.Tensor,
    ) -> tf.Tensor:
        '''with teacher forcing'''

        queries_embedding = self.embedding(tf.convert_to_tensor(queries))
        teacher_forcing_embedding = self.embedding(
            tf.convert_to_tensor(teacher_forcing_responses)
        )

        _, *state = self.lstm_encoder(queries_embedding)

        batch_size = tf.shape(queries)[0]
        outputs = tf.zeros(shape=(
            batch_size,             # batch_size
            MAX_LENGTH,        # max time step
            LSTM_UNIT_NUM,          # dimention of hidden state
        )).numpy()

        import pdb; pdb.set_trace()

        for t in range(MAX_LENGTH):
            _, *state = self.lstm_decoder(
                tf.expand_dims(
                    teacher_forcing_embedding[:, t, :],
                    1,
                ),
                initial_state=state
            )
            # import pdb; pdb.set_trace()
            outputs[:, t, :] = state[0]     # (state_hidden, state_cell)[0]

        outputs = self.time_distributed_dense(outputs)
        return outputs

    def __call_for_normal(
            self,
            queries: tf.Tensor,
    ) -> tf.Tensor:
        '''doc'''
        pass
    #     # [batch_size, max_length]

    #     outputs = self.embedding(queries)
    #     # [batch_size, max_length, vocabulary_size]

    #     _, *states = self.lstm_encoder(outputs)

    #     go_embedding = self.__indice_to_embedding(self.word_index[GO])

    #     batch_size = tf.shape(queries)[0]
    #     outputs = tf.zeros((
    #         batch_size,         # batch_size
    #         MAX_LENGTH,    # max time step
    #     ))

    #     output = go_embedding

    #     for t in range(MAX_LENGTH):
    #         _, *states = self.lstm_decoder(
    #             output,
    #             initial_state=states,
    #         )
    #         output = self.dense(states[0])  # (hidden, cell)[0]
    #         # [self.vocabulary_size]

    #         outputs[:, t] = output
    #         if tf.argmax(output) == self.word_index[DONE]:
    #             break

    #     return outputs

    def predict(self, inputs: List[int], temperature=1.) -> List[int]:
        '''doc'''

        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.expand_dims(inputs, axis=0)

        outputs = self.embedding(inputs)

        _, *states = self.lstm_encoder(outputs)

        output = self.__indice_to_embedding(self.word_index[GO])

        outputs = np.zeros((MAX_LENGTH,))
        for t in range(MAX_LENGTH):
            output, *states = self.lstm_decoder(output, initial_state=states)
            output = self.dense(output[-1])     # last output

            # align the embedding value
            indice = self.__logit_to_indice(output, temperature=temperature)
            output = self.__indice_to_embedding(indice)

            outputs[t] = indice

            if indice == self.word_index[DONE]:
                break

        return outputs

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
        indice = np.random.choice(self.vocabulary_size, p=prob)
        return indice

    def __indice_to_embedding(self, indice: int) -> tf.Tensor:
        tensor = tf.convert_to_tensor([[indice]])
        return self.embedding(tensor)
