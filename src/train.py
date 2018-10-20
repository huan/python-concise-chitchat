"""doc"""
import pickle
from typing import (
    Dict,
    # Tuple,
    # List,
)

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer: tf.keras.preprocessing.text.Tokenizer = pickle.load(f)

VOCABULARY_SIZE = tokenizer.num_words
MAX_LEN = 20
LSTM_UNITS_NUM = 300

# load the data
question_sequence = np.load('data/questions.npy')
answer_sequence = np.load('data/answers.npy')


def load_glove_dict(file: str) -> Dict[str, np.array]:
    '''doc'''
    new_glove_dict: Dict[str, np.array] = {}
    with open(file, 'r') as glove_file:
        for line in glove_file:
            item_list = line.strip().split()
            word = item_list[0]
            embedding = np.array(item_list[1:], dtype=np.float32)
            new_glove_dict[word] = embedding

    return new_glove_dict


glove_dict = load_glove_dict('data/glove.6B.50d.txt')
EMBEDDING_DIMENTION = 50

print('got glove embedding for {} words'.format(len(glove_dict.keys())))
# print('glove embedding for "...": {}'.format(glove_dict.get('...')))

embedding_matrix = np.zeros((VOCABULARY_SIZE, EMBEDDING_DIMENTION))
for index in range(1, tokenizer.num_words):
    word = tokenizer.index_word[index]
    # print('get glove embedding for: {}'.format(word))
    embedding_matrix[index, :] = glove_dict.get(word)

# Use teacher forcing
teacher_forcing_answers = np.zeros(
    (
        answer_sequence.shape[0],
        MAX_LEN,
        VOCABULARY_SIZE,
    ),
    dtype=np.float16,
)
for i, word_id_list in enumerate(answer_sequence):
    for j, word_id in enumerate(word_id_list):
        if j > 0:   # Skip the 'BOS'
            teacher_forcing_answers[i, j - 1, word_id] = 1
    if i % 5000 == 0:
        print('{} entries completed'.format(i))

padded_question_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    question_sequence,
    maxlen=MAX_LEN,
    dtype=np.uint16,
    padding='post',
    truncating='post',
)
padded_answer_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    answer_sequence,
    maxlen=MAX_LEN,
    dtype=np.uint16,
    padding='post',
    truncating='post',
)

embedding_layer = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIMENTION,
    trainable=False,
)
embedding_layer.build((None,))
embedding_layer.set_weights([embedding_matrix])

encoder_lstm_layer = tf.keras.layers.LSTM(
    units=LSTM_UNITS_NUM,
    dropout=0.1,
    recurrent_dropout=0.1,
    return_state=True,
)
decoder_lstm_layer = tf.keras.layers.LSTM(
    units=LSTM_UNITS_NUM,
    dropout=0.1,
    recurrent_dropout=0.1,
    return_state=True,
    return_sequences=True,
)

dense_layer = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(
        VOCABULARY_SIZE,
        activation='softmax',
    )
)

question_input_layer = tf.keras.layers.Input(
    shape=(MAX_LEN, ),
    dtype=np.uint16,
    name='question',
)

answer_input_layer = tf.keras.layers.Input(
    shape=(MAX_LEN, ),
    dtype=np.uint16,
    name='answer',
)

question_embedding_layer = embedding_layer(question_input_layer)
answer_embedding_layer = embedding_layer(answer_input_layer)

encoded_state, hidden_state, cell_state = encoder_lstm_layer(
    question_embedding_layer,
)

decoded_state, _, _ = decoder_lstm_layer(
    answer_embedding_layer,
    initial_state=[
        hidden_state,
        cell_state,
    ],
)

decoded_output = dense_layer(decoded_state)

model = tf.keras.Model(
    [
        question_input_layer,
        answer_input_layer,
    ],
    decoded_output,
)

adam = tf.train.AdamOptimizer(learning_rate=1e-4)
adam_with_clip_norm = tf.contrib.estimator.clip_gradients_by_norm(
    optimizer=adam,
    clip_norm=5,
)


model.compile(
    optimizer=adam_with_clip_norm,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print('Start training ...')
model.fit(
    x=[
        padded_question_sequence,
        padded_answer_sequence,
    ],
    y=teacher_forcing_answers,
    epochs=10,
    batch_size=16,
)
