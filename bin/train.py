'''train'''
import tensorflow as tf

from chit_chat import (
    ChitChat,
    DataLoader,
    Vocabulary,
)

tf.enable_eager_execution()


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

    # import pdb; pdb.set_trace()
    return loss_value


def grad(model, inputs, targets):
    '''doc'''
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    gradients = tape.gradient(loss_value, model.variables)

    clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 50.0)
    # print('gradients: ', gradients)
    # print('clipped_grads: ', clipped_grads)
    # print('global norm: ', global_norm.numpy())

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('global_norm', global_norm.numpy())

    return clipped_grads


def train() -> int:
    '''doc'''
    learning_rate = 1e-2
    num_batches = 8000
    batch_size = 4

    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)
    chitchat = ChitChat(vocabulary=vocabulary)

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

        queries_sequences = vocabulary.texts_to_padded_sequences(
            queries,
            padding='pre',
        )
        responses_sequences = vocabulary.texts_to_padded_sequences(responses)

        grads = grad(chitchat, queries_sequences, responses_sequences)

        optimizer.apply_gradients(
            grads_and_vars=zip(grads, chitchat.variables)
        )

        if step % 10 == 0:
            monitor(
                chitchat,
                vocabulary,
                queries[:3],
                queries_sequences[:3],
                step,
                loss(chitchat, queries_sequences, responses_sequences).numpy()
            )

        if step % 100 == 0:
            checkpoint.save('./data/save/model.ckpt')
            print('checkpoint saved.')

        with tf.contrib.summary.always_record_summaries():
            # your model code goes here
            tf.contrib.summary.scalar('loss', loss(
                chitchat, queries_sequences, responses_sequences).numpy())
            # print('summary had been written.')

        kernel, recurrent_kernel, bias = chitchat.encoder.lstm_encoder.variables
        with tf.name_scope('encoder/kernel'):
            variable_summaries(kernel)
        with tf.name_scope('encoder/recurrent_kernel'):
            variable_summaries(recurrent_kernel)
        with tf.name_scope('encoder/bias'):
            variable_summaries(bias)

        kernel, recurrent_kernel, bias = chitchat.decoder.lstm_decoder.variables
        with tf.name_scope('decoder/kernel'):
            variable_summaries(kernel)
        with tf.name_scope('decoder/recurrent_kernel'):
            variable_summaries(recurrent_kernel)
        with tf.name_scope('decoder/bias'):
            variable_summaries(bias)

        with tf.name_scope('embedding'):
            variable_summaries(chitchat.embedding.variables)

    return 0


def main() -> int:
    '''doc'''
    return train()


def monitor(
        chitchat: ChitChat,
        vocabulary: Vocabulary,
        queries,
        query_sequences,
        step,
        loss_value,
) -> None:
    predicts = chitchat(
        query_sequences,
    )

    predict_sequences = tf.argmax(predicts, axis=2).numpy()

    predict_responses = [
        ' '.join([
            vocabulary.tokenizer.index_word[i]
            for i in predict_sequence
            if i != 0
        ])
        for predict_sequence in predict_sequences
    ]

    print('------- step %d , loss %f -------' % (step, loss_value))
    for query, predict_response in zip(queries, predict_responses):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> %s' % query)
        print('> %s' % predict_response)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # import pdb; pdb.set_trace()
    with tf.contrib.summary.always_record_summaries():
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.contrib.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.contrib.summary.scalar('stddev', stddev)
            tf.contrib.summary.scalar('max', tf.reduce_max(var))
            tf.contrib.summary.scalar('min', tf.reduce_min(var))
            tf.contrib.summary.histogram('histogram', var)


main()
