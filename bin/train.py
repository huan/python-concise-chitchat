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
    predictions = model(
        inputs=x,
        teacher_forcing_targets=y,
        training=True,
    )

    # implment the following contrib function in a loop ?
    # https://stackoverflow.com/a/41135778/1123955
    # https://stackoverflow.com/q/48025004/1123955
    # loss_value = tf.contrib.seq2seq.sequence_loss(
    #     prediction,
    #     tf.convert_to_tensor(y),
    #     weights,
    # )

    # import pdb; pdb.set_trace()

    y_without_go = tf.concat(
        [
            y[:, 1:],
            tf.expand_dims(tf.zeros(tf.shape(y)[0], dtype=tf.int32), axis=1)
        ],
        axis=1
    ).numpy()

    weights = tf.cast(
        tf.not_equal(y_without_go, 0),
        tf.float32,
    )

    # https://stackoverflow.com/a/45823378/1123955
    t = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_without_go,
            logits=predictions,
        )

    loss_value = tf.reduce_sum(
        t * weights
    )
    loss_value = loss_value / tf.cast(tf.shape(y)[0], tf.float32)

    if True:
        print('predictions: ', [
            indice
            for indice in tf.argmax(predictions[0], axis=1).numpy()
        ])
        print('%s [%s]' % ('y_without_go:', ', '.join([str(i) for i in y_without_go[0]])))
        print('loss: ', loss_value.numpy())

    # import pdb; pdb.set_trace()
    return loss_value


def grad(model, inputs, targets):
    '''doc'''
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    gradients = tape.gradient(loss_value, model.variables)

    # clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 50.0)
    # print('gradients: ', gradients)
    # print('clipped_grads: ', clipped_grads)
    # print('global norm: ', global_norm.numpy())

    return gradients


def train() -> int:
    '''doc'''
    learning_rate = 1e-2
    num_batches = 8000
    batch_size = 64

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

        if step % 10 == 0:
            print("step %d: loss %f" % (step, loss(
                chitchat, queries_sequences, responses_sequences).numpy()))
            checkpoint.save('./data/save/model.ckpt')
            print('checkpoint saved.')

            # print('query: %s' % queries[0])
            print('response: %s' % responses[0])
            predicts = chitchat(
                queries_sequences,
                # teacher_forcing_targets=responses_sequences,
                # training=True,
            )

            predict_sequence = tf.argmax(predicts[0], axis=1).numpy()
            # print('predict sequence: %s' % predict_sequence)

            predict_response = [
                vocabulary.tokenizer.index_word[i]
                for i in predict_sequence
                if i != 0
            ]
            print('predict response: %s' % predict_response)

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
