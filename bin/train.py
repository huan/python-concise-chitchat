'''train'''
import re

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from config import (
    DONE,
    GO,
    MAX_LENGTH,
)
from data_loader import DataLoader
from model import ChitChat


def train() -> int:
    '''doc'''
    ####################
    learning_rate = 1e-3
    num_batches = 10
    batch_size = 128

    data_loader = DataLoader()
    # vocabulary = Vocabulary(data_loader.raw_text)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(
        re.split(
            r'[\s\t\n]',
            data_loader.raw_text,
        ) + [GO, DONE]
    )

    chitchat = ChitChat(tokenizer.word_index)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    for batch_index in range(num_batches):
        batch_query, batch_response = data_loader.get_batch(batch_size)

        encoder_input = pad_seq([
            tokenizer.texts_to_sequences(query)
            for query in batch_query
        ])
        decoder_input = pad_seq([
            tokenizer.texts_to_sequences(response)
            for response in batch_response
        ])
        decoder_target = pad_seq([
            tokenizer.texts_to_sequences(
                response[1:]    # get rid of the start GO
            )
            for response in batch_response
        ])

        with tf.GradientTape() as tape:
            sequence_logit_pred = chitchat(
                inputs=encoder_input,
                teacher_forcing_inputs=decoder_input,
                training=True,
            )

            # implment the following contrib function in a loop ?
            # https://stackoverflow.com/a/41135778/1123955
            # https://stackoverflow.com/q/48025004/1123955
            loss = tf.contrib.seq2seq.sequence_loss(
                sequence_logit_pred,
                decoder_target,
                tf.ones_like(decoder_target),
            )
            print("batch %d: loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, chitchat.variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(grads, chitchat.variables)
        )

    return 0


def pad_seq(seq) -> tf.Tensor:
    '''doc'''
    seq = np.array(seq)
    return tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        dtype=tf.uint16,
        padding='post',
        truncating='post',
    )


def main() -> int:
    '''doc'''
    return train()


main()
