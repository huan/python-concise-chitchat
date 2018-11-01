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
    loss_value = tf.contrib.seq2seq.sequence_loss(
        prediction,
        tf.convert_to_tensor(y),
        weights,
    )

    print('prediction: ', [indice for indice in tf.argmax(prediction[0], axis=1).numpy()])
    print('y: ', y[0])
    print('loss: ', loss_value.numpy())

    # import pdb; pdb.set_trace()
    return loss_value


def grad(model, inputs, targets):
    '''doc'''
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return tape.gradient(loss_value, model.variables)


def train() -> int:
    '''doc'''
    learning_rate = 1e-2
    num_batches = 8000
    batch_size = 32

    print('Dataset size: {}, Vocabulary size: {}'.format(
        data_loader.size,
        vocabulary.size,
    ))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=chitchat,
        optimizer_step=tf.train.get_or_create_global_step(),
    )

    checkpoint.restore(tf.train.latest_checkpoint('./data/save'))
    print('checkpoint restored.')

    writer = tf.contrib.summary.create_file_writer('./data/board')
    writer.set_as_default()

    global_step = tf.train.get_or_create_global_step()

    for step in range(num_batches):
        global_step.assign_add(1)

        queries, responses = data_loader.get_batch(batch_size)

        queries_sequences = vocabulary.texts_to_padded_sequences(queries)
        responses_sequences = vocabulary.texts_to_padded_sequences(responses)

        grads = grad(chitchat, queries_sequences, responses_sequences)

        optimizer.apply_gradients(
            grads_and_vars=zip(grads, chitchat.variables)
        )

        if step % 10 == 0 or True:
            print("step %d: loss %f" % (step, loss(
                chitchat, queries_sequences, responses_sequences).numpy()))
            checkpoint.save('./data/save/model.ckpt')
            print('checkpoint saved.')

        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            # your model code goes here
            tf.contrib.summary.scalar('loss', loss(
                chitchat, queries_sequences, responses_sequences).numpy())
            # print('summary had been written.')

    return 0


def main() -> int:
    '''doc'''
    return train()


main()
