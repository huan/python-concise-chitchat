'''doc'''

import tensorflow as tf
import numpy as np
# from typing import (
#     List,
#     Tuple,
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
            decoder_inputs: tf.Tensor=None,
            training=None,
    ) -> tf.Tensor:
        '''call'''

        if not training:
            return self.normal_call(inputs)

        if decoder_inputs is None:
            raise ValueError('decoder_inputs not set when training')

        return self.training_call(inputs, decoder_inputs)

    def training_call(
            self,
            questions: tf.Tensor,
            answers: tf.Tensor,
    ) -> tf.Tensor:
        '''with teacher forcing'''
        questions_embedding = self.embedding(questions)
        answers_embedding = self.embedding(answers)

        _, *state = self.lstm_encoder(questions_embedding)

        outputs = tf.zeros(
            (
                tf.shape(questions)[0],
                self.max_length,
            ),
            dtype=tf.bfloat16,
        )

        for t in range(self.max_length):
            output, *state = self.lstm_decoder(
                answers_embedding[:, t, :],
                initial_state=state
            )
            outputs[:, t, :] = output

        outputs = self.time_distributed_dense(outputs)
        return outputs

    def normal_call(
            self,
            inputs: tf.Tensor,
    ) -> tf.Tensor:
        # inputs: [batch_size, max_length]

        outputs = self.embedding(inputs)
        # outputs: [batch_size, max_length, vocabulary_size]

        _, *state = self.lstm_encoder(outputs)

        start_token_embedding = self.embedding([[[
            self.vocabulary.start_token_index
        ]]]).numpy().flatten()

        outputs = tf.zeros(
            (
                tf.shape(inputs)[0],  # batch_size
                self.max_length,
            ),
            dtype=tf.bfloat16,
        )

        output = start_token_embedding
        for t in range(self.max_length):
            output, *state = self.lstm_decoder(
                output,
                initial_state=state,
            )
            outputs[:, t, :] = output

        outputs = self.time_distributed_dense(outputs)
        return outputs

    def predict(self, inputs, state=None, temperature=1.):
        '''predict'''
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([
            np.random.choice(self.vocabulary.size, p=prob[i, :])
            for i in range(batch_size.numpy())
        ])


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
        batch_sentence, batch_next_sentence = data_loader.get_batch(batch_size)

        encoder_input = [
            vocabulary.sentence_to_sequence(
                [vocabulary.start_token]
                    + sentence
                    + [vocabulary.end_token]
            )
            for sentence in batch_sentence
        ]
        decoder_input = [
            vocabulary.sentence_to_sequence(
                [vocabulary.start_token]
                + next_sentence
            )
            for next_sentence in batch_next_sentence
        ]
        decoder_target = [
            vocabulary.sentence_to_sequence(
                next_sentence
                + [vocabulary.end_token]
            )
            for next_sentence in batch_next_sentence
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
