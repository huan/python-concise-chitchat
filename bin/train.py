'''train'''
import tensorflow as tf

from chit_chat import (
    ChitChat,
    DataLoader,
    Vocabulary,
)

tf.enable_eager_execution()

data_loader = DataLoader()
vocabulary = Vocabulary(data_loader.raw_text)
chitchat = ChitChat(vocabulary=vocabulary)


def loss(model, x, y) -> tf.Tensor:
    '''doc'''
    weights = tf.cast(
        tf.not_equal(y, 0),
        tf.float32,
    )

    prediction = model(
        inputs=x,
        teacher_forcing_targets=y,
        training=True,
    )

    # implment the following contrib function in a loop ?
    # https://stackoverflow.com/a/41135778/1123955
    # https://stackoverflow.com/q/48025004/1123955
    return tf.contrib.seq2seq.sequence_loss(
        prediction,
        tf.convert_to_tensor(y),
        weights,
    )


def grad(model, inputs, targets):
    '''doc'''
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return tape.gradient(loss_value, model.variables)


def train() -> int:
    '''doc'''
    learning_rate = 1e-3
    num_batches = 8000
    batch_size = 1024

    print('Dataset size: {}, Vocabulary size: {}'.format(
        data_loader.size,
        vocabulary.size,
    ))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    root = tf.train.Checkpoint(
        optimizer=optimizer,
        model=chitchat,
        optimizer_step=tf.train.get_or_create_global_step(),
    )

    root.restore(tf.train.latest_checkpoint('./data/save'))
    print('checkpoint restored.')

    writer = tf.contrib.summary.create_file_writer('./data/board')
    writer.set_as_default()

    global_step = tf.train.get_or_create_global_step()

    for batch_index in range(num_batches):
        global_step.assign_add(1)

        queries, responses = data_loader.get_batch(batch_size)

        encoder_inputs = vocabulary.texts_to_padded_sequences(queries)
        decoder_outputs = vocabulary.texts_to_padded_sequences(responses)

        grads = grad(chitchat, encoder_inputs, decoder_outputs)

        optimizer.apply_gradients(
            grads_and_vars=zip(grads, chitchat.variables)
        )

        if batch_index % 10 == 0:
            print("batch %d: loss %f" % (batch_index, loss(
                chitchat, encoder_inputs, decoder_outputs).numpy()))
            root.save('./data/save/model.ckpt')
            print('checkpoint saved.')

        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            # your model code goes here
            tf.contrib.summary.scalar('loss', loss(
                chitchat, encoder_inputs, decoder_outputs).numpy())
            # print('summary had been written.')

    return 0


def main() -> int:
    '''doc'''
    return train()


main()
