'''train'''
import re

import tensorflow as tf

from chit_chat import (
    DataLoader,
    ChitChat,
    DONE,
    GO,
    MAX_LENGTH,
)

tf.enable_eager_execution()

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')


def train() -> int:
    '''doc'''
    ####################
    learning_rate = 1e-3
    num_batches = 4000
    batch_size = 64

    data_loader = DataLoader()
    # vocabulary = Vocabulary(data_loader.raw_text)

    tokenizer.fit_on_texts(
        [GO, DONE] + re.split(
            r'[\s\t\n]',
            data_loader.raw_text,
        )
    )

    print('Dataset size: {}, Vocabulari size: {}'.format(
        data_loader.size,
        len(tokenizer.word_index.keys()),
    ))
    chitchat = ChitChat(tokenizer.word_index)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    for batch_index in range(num_batches):
        queries, responses = data_loader.get_batch(batch_size)

        encoder_input = texts_to_padded_sequences(queries)

        decoder_input = texts_to_padded_sequences(responses)

        decoder_target = tf.concat(
            (
                decoder_input[:, 1:],   # get rid of the start GO
                tf.zeros((batch_size, 1), dtype=tf.int32),
            ),
            axis=-1,
        )

        weights = tf.cast(
            tf.not_equal(decoder_target, 0),
            tf.float32,
        )

        # import pdb; pdb.set_trace()

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
                weights,
            )
            # TODO: mask length ?
            print("batch %d: loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, chitchat.variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(grads, chitchat.variables)
        )

    return 0


def texts_to_padded_sequences(text: str) -> tf.Tensor:
    '''doc'''
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post',
    )

    # return tf.convert_to_tensor(padded_sequences)
    return padded_sequences


def main() -> int:
    '''doc'''
    return train()


main()
