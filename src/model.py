'''doc'''

import tensorflow as tf
import numpy as np
# from typing import (
#     List,
#     # Tuple,
# )

from dataloader import DataLoader
from vocabulary import Vocabulary

EMBEDDING_DIMENTION = 100
LSTM_UNIT_NUM = 300
LSTM_DROPOUT = 0.1
LSTM_RECURRENT_DROPOUT = 0.1


class ChitChat(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary: Vocabulary,
            max_length: int,
    ) -> None:
        super().__init__()

        self.max_length = max_length
        self.vocabulary = vocabulary

        # [batch_size, max_length] -> [batch_size, max_length, vocabulary_size]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocabulary.size,
            output_dim=EMBEDDING_DIMENTION,
            mask_zero=True,
            name='embedding',
        )
        self.lstm_encoder = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            name='lstm_encoder'
        )
        self.lstm_decoder = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            return_sequences=True,
            name='lstm_decoder',
        )

        self.dense = tf.keras.layers.Dense(
            units=self.vocabulary.size,
            activation='softmax',
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
        queries_embedding = self.embedding(queries)
        teacher_forcing_embedding = self.embedding(teacher_forcing_responses)

        _, *state = self.lstm_encoder(queries_embedding)

        batch_size = tf.shape(queries)[0]
        outputs = tf.zeros(shape=(
            batch_size,             # batch_size
            self.max_length,        # max time step
            LSTM_UNIT_NUM,          # dimention of hidden state
        ))

        for t in range(self.max_length):
            _, *state = self.lstm_decoder(
                teacher_forcing_embedding[:, t, :],
                initial_state=state
            )
            outputs[:, t, :] = state[0]     # (state_hidden, state_cell)[0]

        outputs = self.time_distributed_dense(outputs)
        return outputs

    def __call_for_normal(
            self,
            queries: tf.Tensor,
    ) -> tf.Tensor:
        '''doc'''
        # [batch_size, max_length]

        outputs = self.embedding(queries)
        # [batch_size, max_length, vocabulary_size]

        _, *states = self.lstm_encoder(outputs)

        start_token_embedding = self.embedding([[
            self.vocabulary.start_token_indice
        ]]).numpy().flatten()

        batch_size = tf.shape(queries)[0]
        outputs = tf.zeros((
            batch_size,         # batch_size
            self.max_length,    # max time step
        ))

        output = start_token_embedding

        for t in range(self.max_length):
            _, *states = self.lstm_decoder(
                output,
                initial_state=states,
            )
            output = self.dense(states[0])  # (hidden, cell)[0]
            # [self.vocabulary.size]

            outputs[:, t] = output
            if tf.argmax(output) == self.vocabulary.end_token_indice:
                break

        return outputs

    def predict(self, inputs: tf.Tensor, temperature=1.) -> tf.Tensor:
        '''
        inputs: queries [1, max_length]
        outputs: responses [1, max_length]
        '''
        outputs = self.embedding(inputs)

        _, *states = self.lstm_encoder(outputs)

        output = self.__indice_to_embedding(self.vocabulary.start_token_indice)

        outputs = np.zeros((self.max_length,))
        for t in range(self.max_length):
            output, *states = self.lstm_decoder(output, initial_state=states)
            output = self.dense(states[0])

            # align the embedding value
            indice = self.__logit_to_indice(output, temperature=temperature)
            output = self.__indice_to_embedding(indice)

            outputs[t] = indice

            if indice == self.vocabulary.end_token_indice:
                break

    def __logit_to_indice(
            self,
            inputs,
            temperature=1.,
    ) -> int:
        '''
        [vocabulary_size]
        convert one hot encoding to indice with temperature
        '''
        prob = tf.nn.softmax(inputs / temperature).numpy()
        indice = np.random.choice(self.vocabulary.size, p=prob)
        return indice

    def __indice_to_embedding(self, indice: int) -> tf.Tensor:
        embedding = self.embedding([[indice]]).numpy().flatten()
        return embedding



def train() -> int:
    '''doc'''
    ####################
    learning_rate = 1e-3
    num_batches = 10
    batch_size = 128
    max_length = 20

    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)

    chitchat = ChitChat(
        max_length=max_length,
        vocabulary=vocabulary,
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    for batch_index in range(num_batches):
        batch_query, batch_response = data_loader.get_batch(batch_size)

        encoder_input = [
            vocabulary.sentence_to_sequence(
                [vocabulary.start_token]
                    + sentence
                    + [vocabulary.end_token]
            )
            for sentence in batch_query
        ]
        decoder_input = [
            vocabulary.sentence_to_sequence(
                [vocabulary.start_token]
                + next_sentence
            )
            for next_sentence in batch_response
        ]
        decoder_target = [
            vocabulary.sentence_to_sequence(
                next_sentence
                + [vocabulary.end_token]
            )
            for next_sentence in batch_response
        ]

        with tf.GradientTape() as tape:
            sequence_logit_pred = chitchat(encoder_input)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=sequence_logit_pred)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, chitchat.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, chitchat.variables))

    return 0

def inference() -> None:
    '''inference'''
    X_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            y_pred = chitchat.predict(X, diversity)
            print(data_loader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)

    return
