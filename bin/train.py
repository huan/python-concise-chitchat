'''
train
'''
import tensorflow as tf

from chit_chat import (
    BATCH_SIZE,
    MAX_LEN,
    EOS,
    LEARNING_RATE,
    PAD,

    ChitChat,
    DataLoader,
    Vocabulary,
)

# tf.enable_eager_execution()

data_loader = DataLoader()
vocabulary = Vocabulary(data_loader.raw_text)
chitchat = ChitChat(vocabulary=vocabulary)

PAD_INDICE = vocabulary.tokenizer.word_index.get(PAD)

# chitchat([[1,2,3,4,5,6,7,8,9,0, 1,2,3,4,5,6,7,8,9,0]])
# print('###################################')
# print(chitchat.summary())


def loss_function(
        model,
        x,
        y,
) -> tf.Tensor:
    '''doc'''
    # import pdb; pdb.set_trace()

    predictions = model(
        inputs=x,
        training=True,
        teacher_forcing_targets=y,
    )

    # import pdb; pdb.set_trace()

    y_without_sos = tf.concat(
        [
            y[:, 1:],
            tf.expand_dims(tf.fill([BATCH_SIZE], PAD_INDICE), axis=1)
        ],
        axis=1
    )

    mask = tf.not_equal(y_without_sos, PAD_INDICE)
    mask = tf.cast(mask, tf.float32)

    # https://stackoverflow.com/a/45823378/1123955
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_without_sos,
        logits=predictions,
    )
    loss = loss * mask

    loss_value = tf.reduce_sum(loss)
    loss_value = loss_value / tf.reduce_sum(mask)

    return loss_value


def grad(model, inputs, targets):
    '''doc'''
    with tf.GradientTape() as tape:
        loss_value = loss_function(model, inputs, targets)

    gradients = tape.gradient(loss_value, model.variables)

    clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 5.)
    # print('gradients: ', gradients)
    # print('clipped_grads: ', clipped_grads)
    # print('global norm: ', global_norm.numpy())

    # with tf.summary.always_record_summaries():
    tf.summary.scalar('global_norm', global_norm.numpy(), step=tf.compat.v1.train.get_or_create_global_step())

    return clipped_grads


def train() -> int:
    '''doc'''
    num_steps = 500000000

    print('Dataset size: {}, Vocabulary size: {}'.format(
        data_loader.size,
        vocabulary.size,
    ))

    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=chitchat,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step(),
    )

    checkpoint.restore(tf.train.latest_checkpoint('./data/save'))
    print('checkpoint restored.')

    writer = tf.summary.create_file_writer('./data/board')
    writer.set_as_default()

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # print(chitchat.summary())

    for step in range(num_steps):
        global_step.assign_add(1)

        queries, responses = data_loader.get_batch(BATCH_SIZE)

        queries_sequences = vocabulary.texts_to_padded_sequences(queries)
        responses_sequences = vocabulary.texts_to_padded_sequences(responses)

        grads = grad(chitchat, queries_sequences, responses_sequences)

        optimizer.apply_gradients(
            grads_and_vars=zip(grads, chitchat.variables)
        )

        if step % 10 == 0:
            monitor(
                chitchat,
                vocabulary,
                queries[:2],
                responses[:2],
                queries_sequences[:2],
                tf.compat.v1.train.get_or_create_global_step(),
                loss_function(chitchat, queries_sequences, responses_sequences).numpy()
            )

        if step % 100 == 0:
            checkpoint.save('./data/save/model.ckpt')
            print('checkpoint saved.')

        # your model code goes here
        tf.summary.scalar(
            'loss',
            loss_function(
                chitchat, queries_sequences, responses_sequences
            ).numpy(),
            step=tf.compat.v1.train.get_or_create_global_step(),
        )
        # print('summary had been written.')

    return 0


def main() -> int:
    '''doc'''
    return train()


def monitor(
        chitchat: ChitChat,
        vocabulary: Vocabulary,
        queries,
        responses,
        query_sequences,
        step,
        loss_value,
) -> None:
    '''doc'''
    # kernel, recurrent_kernel, bias = chitchat.encoder.lstm_encoder.variables
    # with tf.name_scope('encoder/kernel'):
    #     variable_summaries(kernel)
    # with tf.name_scope('encoder/recurrent_kernel'):
    #     variable_summaries(recurrent_kernel)
    # with tf.name_scope('encoder/bias'):
    #     variable_summaries(bias)

    # kernel, recurrent_kernel, bias = chitchat.decoder.lstm_decoder.variables
    # with tf.name_scope('decoder/kernel'):
    #     variable_summaries(kernel)
    # with tf.name_scope('decoder/recurrent_kernel'):
    #     variable_summaries(recurrent_kernel)
    # with tf.name_scope('decoder/bias'):
    #     variable_summaries(bias)

    with tf.name_scope('embedding'):
        variable_summaries(chitchat.embedding.variables)

    # output

    predicts = chitchat(
        query_sequences,
    )

    predict_sequences = tf.argmax(predicts, axis=2).numpy()

    EOS_INDICE = vocabulary.tokenizer.word_index.get(EOS)

    # import pdb; pdb.set_trace()

    predict_responses = [
        ' '.join([
            vocabulary.tokenizer.index_word[i]
            for i in predict_sequence
            if i != EOS_INDICE
        ])
        for predict_sequence in predict_sequences
    ]

    print('------- step %d , loss %f -------' % (step, loss_value))
    for query, response, predict_response \
            in zip(queries, responses, predict_responses):
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< %s' % query)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> %s' % response)
        print('> %s\n' % predict_response)


def variable_summaries(var):
    '''Attach a lot of summaries to a Tensor (for visualization).'''
    # import pdb; pdb.set_trace()
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, step=tf.compat.v1.train.get_or_create_global_step())
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, step=tf.compat.v1.train.get_or_create_global_step())
        tf.summary.scalar('max', tf.reduce_max(var), step=tf.compat.v1.train.get_or_create_global_step())
        tf.summary.scalar('min', tf.reduce_min(var), step=tf.compat.v1.train.get_or_create_global_step())
        tf.summary.histogram('histogram', var, step=tf.compat.v1.train.get_or_create_global_step())


main()
