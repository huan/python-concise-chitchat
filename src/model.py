'''doc'''

import tensorflow as tf
import numpy as np
from typing import (
    List,
    Tuple,
)

from dataloader import DataLoader
from vocabulary import Vocabulary

EMBEDDING_DIMENTION = 100
LSTM_UNIT_NUM = 300
LSTM_DROPOUT = 0.1
LSTM_RECURRENT_DROPOUT = 0.1


class QuestionEncoder(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
            # max_length: int,
    ) -> None:
        super().__init__()
        # self.max_length = max_length

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=EMBEDDING_DIMENTION,
            name='embedding',
        )
        self.encoder_lstm = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            name='encoder_lstm'
        )

    def call(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        inputs: [batch_size, max_length]
        '''
        output = self.embedding(inputs)
        _, state_hidden, state_cell = self.encoder_lstm(output)
        return state_hidden, state_cell


class AnswerDecoder(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.decoder_lstm = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            return_sequences=True,
            name='decoder_lstm',
        )

        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=vocabulary_size,
                activation='softmax',
            )
        )

    def call(
            self,
            inputs,
            initial_state=None
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        '''
        initial_state = [state_hidden, state_cell]
        '''
        output, state_hidden, state_cell = self.decoder_lstm(
            inputs,
            initial_state=initial_state,
        )
        next_state = [state_hidden, state_cell]

        output = self.dense(output)

        return output, next_state


class ChitChat(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
            max_length: int,
    ) -> None:
        super().__init__()

        self.max_length = max_length
        self.vocabulary_size = vocabulary_size

        self.question_encoder = QuestionEncoder(vocabulary_size)
        self.answer_decoder = AnswerDecoder(vocabulary_size)

    def call(
            self,
            inputs: tf.Tensor,
            decoder_inputs: tf.Tensor=None,
            training=None,
    ) -> tf.Tensor:
        '''call'''

        if (training):
            if decoder_inputs is None:
                raise ValueError('decoder_inputs not set when training')

            initial_state = self.question_encoder(inputs)
            output, = self.answer_decoder(decoder_inputs, initial_state)

            return output

        # not training

        state_hidden, state_cell = self.question_encoder(inputs)
        state = [state_hidden, state_cell]

        outputs = np.zeros(
            (
                tf.shape(inputs)[0],  # batch_size
                self.max_length,
                self.vocabulary_size
            ),
            dtype=tf.bfloat16,
        )

        for t in range(self.max_length):
            state_decoded, state = self.answer_decoder(inputs[:, t, :], state)
            outputs[:, t, :] = state_decoded

        return outputs

    def predict(self, inputs, temperature=1.):
        '''predict'''
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([
            np.random.choice(self.vocabulary_size, p=prob[i, :])
            for i in range(batch_size.numpy())
        ])


def train() -> int:
    '''doc'''
    ####################
    learning_rate = 1e-3
    num_batches = 10
    batch_size = 128

    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)

    model = ChitChat(
        vocabulary_size=data_loader.size+1,
        embedding_dimention=EMBEDDING_DIMENTION,
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    for batch_index in range(num_batches):
        sentence, next_sentence = data_loader.get_batch(batch_size)

        sequence = vocabulary.sentence_to_seqence(sentence)
        next_sequence = vocabulary.sentence_to_seqence(next_sentence)

        with tf.GradientTape() as tape:
            word_logit_pred = model(sequence)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=word_logit_pred)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return 0

def inference() -> None:
    '''inference'''
    X_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            y_pred = model.predict(X, diversity)
            print(data_loader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)

    return
