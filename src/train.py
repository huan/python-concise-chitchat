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

# load the data
current_sentence_sequence = np.load('data/current_sentence.npy')
next_sentence_sequence = np.load('data/next_sentence.npy')

with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer: tf.keras.preprocessing.text.Tokenizer = pickle.load(f)

VOCABULARY_SIZE = tokenizer.num_words
MAX_LEN = 20


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
GLOVE_DIMENTION = 50

print('got glove embedding for {} words'.format(len(glove_dict.keys())))
# print('glove embedding for "...": {}'.format(glove_dict.get('...')))

embedding_matrix = np.zeros((VOCABULARY_SIZE, GLOVE_DIMENTION))
for index in range(1, tokenizer.num_words):
    word = tokenizer.index_word[index]
    # print('get glove embedding for: {}'.format(word))
    embedding_matrix[index, :] = glove_dict.get(word)

# Use teacher forcing
teacher_forcing_sentence = np.zeros((
    next_sentence_sequence.shape[0],
    MAX_LEN,
    VOCABULARY_SIZE,
))
for i, word_id_list in enumerate(next_sentence_sequence):
    for j, word_id in enumerate(word_id_list):
        if j > 0:   # Skip the 'BOS'
            teacher_forcing_sentence[i, j - 1, word_id] = 1
    if i % 5000 == 0:
        print('{} entries completed'.format(i))

padded_current_sentence_list = tf.keras.preprocessing.sequence.pad_sequences(
    current_sentence_sequence,
    maxlen=20,
    dtype='int32',
    padding='post',
    truncating='post',
)
padded_next_sentence_list = tf.keras.preprocessing.sequence.pad_sequences(
    next_sentence_sequence,
    maxlen=MAX_LEN,
    dtype='int32',
    padding='post',
    truncating='post',
)

embedding_layer = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=50,
    trainable=False,
)
embedding_layer.build((None,))
embedding_layer.set_weights([embedding_matrix])

encoder_lstm_layer = tf.keras.layers.LSTM(
    300,
    dropout=0.1,
    recurrent_dropout=0.1,
    return_state=True,
)
decoder_lstm_layer = tf.keras.layers.LSTM(
    300,
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

current_sentence_input_layer = tf.keras.layers.Input(
    shape=(MAX_LEN, ),
    dtype='int32',
    name='current_sentence',
)
next_sentence_input_layer = tf.keras.layers.Input(
    shape=(MAX_LEN, ),
    dtype='int32',
    name='next_sentence',
)

current_sentence_embedding_layer = embedding_layer(current_sentence_input_layer)
next_sentence_embedding_layer = embedding_layer(next_sentence_input_layer)

encoded_state, hidden_state, cell_state = encoder_lstm_layer(
    current_sentence_embedding_layer,
)

decoded_state, _, _ = decoder_lstm_layer(
    next_sentence_embedding_layer,
    initial_state=[
        hidden_state,
        cell_state,
    ],
)

predict_output = dense_layer(decoded_state)

model = tf.keras.Model(
    [
        current_sentence_input_layer,
        next_sentence_input_layer
    ],
    predict_output,
)

adam = tf.train.AdamOptimizer()
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
    [
        padded_current_sentence_list,
        padded_next_sentence_list,
    ],
    teacher_forcing_sentence,
    epochs=10,
    batch_size=16,
)
