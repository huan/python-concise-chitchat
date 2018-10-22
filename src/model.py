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


    def call(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        inputs: [batch_size, max_length]
        '''
        return state_hidden, state_cell


class AnswerDecoder(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
    ) -> None:
        super().__init__()

        self.vocabulary_size = vocabulary_size

    def call(
            self,
            inputs,     # [batch_size, max_length]
            initial_state=None
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        '''
        initial_state = [state_hidden, state_cell]
        '''
)

        state = initial_state

        output, state_hidden, state_cell = self.decoder_lstm(
            inputs_one_hot,
            initial_state=state,
        )
        state = [state_hidden, state_cell]

        output = self.dense(output)

        # convert one_hot encoding to indices number
        output = tf.argmax(output, -1).numpy()

        return output, state


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

        # [batch_size, max_length] -> [batch_size, max_length, vocabulary_size]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=EMBEDDING_DIMENTION,
            name='embedding',
        )
        self.lstm_encoder = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            name='encoder_lstm'
        )
        self.lstm_decoder = tf.keras.layers.LSTM(
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
            inputs: tf.Tensor,
            decoder_inputs: tf.Tensor=None,
            training=None,
    ) -> tf.Tensor:
        '''call'''

        if (training):
            if decoder_inputs is None:
                raise ValueError('decoder_inputs not set when training')

            return self.training_call(inputs, decoder_inputs)
        else:
            return self.normal_call(inputs)

    def training_call(
        self,
        inputs: tf.Tensor,
        decoder_inputs: tf.Tensor,
    ) -> tf.Tensor:
        '''with teacher forcing'''
        question_embedding = self.embedding(inputs)
        answer_embedding = self.embedding(decoder_inputs)

        _, encoder_state_hidden, encoder_state_cell = self.lstm_encoder(question_embedding)

        state = [encoder_state_hidden, encoder_state_cell]
        for t in rang(self.max_length - 1):
            output, state_hidden, state_cell = self.lstm_decoder(
                answer_embedding[:, t, :],
                initial_state=state
            )
            state = [state_hidden, state_cell]
            outputs[:, t + 1, :] = output

        return outputs

    def normal_call(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        # inputs: [batch_size, max_length]

        outputs = self.embedding(inputs)
        # outputs: [batch_size, max_length, vocabulary_size]

        _, encoder_state_hidden, encoder_state_cell = self.lstm_encoder(outputs)

        outputs = np.zeros(
            (
                tf.shape(inputs)[0],  # batch_size
                self.max_length,
                self.vocabulary_size
            ),
            dtype=tf.bfloat16,
        )

        outputs[:, 0, :] = self.vocabulary_START_EMBEDDING

        for t in rang(self.max_length - 1):
            output, decoder_state_hidden, decoder_state_cell = self.decoder_lstm(
                outputs[:, t, :],
                initial_state=[
                    encoder_state_hidden,
                    encoder_state_cell,
                ],
            )
            outputs[:, t + 1, :] = decoder_state_hidden  # output[-1]

        outputs = self.dense(outputs)
        return outputs,

    def predict(self, inputs, state=None, temperature=1.):
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
        max_length=20,
        vocabulary_size=vocabulary.size,
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    for batch_index in range(num_batches):
        batch_sentence, batch_next_sentence = data_loader.get_batch(batch_size)

        encoder_input = [
            vocabulary.sentence_to_sequence(sentence)
            for sentence in batch_sentence
        ]
        decoder_input = [
            vocabulary.sentence_to_sequence(next_sentence)
            for next_sentence in batch_next_sentence
        ]

        with tf.GradientTape() as tape:
            word_logit_pred = model(encoder_input)
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
