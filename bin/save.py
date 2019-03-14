'''Save Model'''

import tensorflow as tf


from chit_chat import (
    ChitChat,
    DataLoader,
    Vocabulary,
    END_TOKEN,
)


def main() -> int:
    '''chat main'''
    data_loader = DataLoader()
    vocabulary = Vocabulary(data_loader.raw_text)

    print('Dataset size: {}, Vocabulary size: {}'.format(
        data_loader.size,
        vocabulary.size,
    ))

    chitchat = ChitChat(vocabulary)
    checkpoint = tf.train.Checkpoint(model=chitchat)
    checkpoint.restore(tf.train.latest_checkpoint('./data/save'))
    print('checkpoint restored.')

    # ([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # print(chitchat.get_config())

    tf.keras.models.save_model(chitchat, 'epic_num_reader.model')
    # tf.contrib.saved_model.save_keras_model(
    #     chitchat,
    #     'data/saved_model/',
    #     serving_only=True,
    # )
    # chitchat.save_weights('chitchat.h5')

    # chitchat.compile(loss='binary_crossentropy', optimizer='sgd')

    # estimator = tf.keras.estimator.model_to_estimator(
    #     keras_model=chitchat,
    #     model_dir='./data/model_dir',
    # )

    # def serving_input_receiver_fn():
    #     return tf.estimator.export.build_raw_serving_input_receiver_fn({
    #         chitchat.input_names[0]: tf.placeholder(tf.float32, shape=[None, 10])
    #     })

    # estimator.export_savedmodel(
    #     './exported',
    #     serving_input_receiver_fn=serving_input_receiver_fn,
    # )

main()

