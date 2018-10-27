'''train'''
import tensorflow as tf

from chit_chat import (
    ChitChat,
    DataLoader,
    Vocabulary,
)

tf.enable_eager_execution()


def train() -> int:
    '''doc'''
    learning_rate = 1e-3
    num_batches = 8000
    batch_size = 256

    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)

    print('Dataset size: {}, Vocabulary size: {}'.format(
        data_loader.size,
        vocabulary.size,
    ))
    chitchat = ChitChat(vocabulary=vocabulary)

    checkpoint = tf.train.Checkpoint(model=chitchat)
    checkpoint.restore(tf.train.latest_checkpoint('./data/save'))
    print('checkpoint restored.')

    summary_writer = tf.contrib.summary.create_file_writer('./data/tensorboard')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():

        for batch_index in range(num_batches):
            queries, responses = data_loader.get_batch(batch_size)

            encoder_input = vocabulary.texts_to_padded_sequences(queries)

            decoder_input = vocabulary.texts_to_padded_sequences(responses)

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

            tf.contrib.summary.scalar("loss", loss, step=batch_index)

            grads = tape.gradient(loss, chitchat.variables)
            optimizer.apply_gradients(
                grads_and_vars=zip(grads, chitchat.variables)
            )

            print("batch %d: loss %f" % (batch_index, loss.numpy()))

            if batch_index % 100 == 0:
                checkpoint.save('./data/save/model.ckpt')
                print('checkpoint saved.')

    return 0


def main() -> int:
    '''doc'''
    return train()


main()
