'''doc'''

import tensorflow as tf

from dataloader import DataLoader
from vocabulary import Vocabulary

EMBEDDING_DIMENTION = 300
LSTM_UNIT_NUM = 300
LSTM_DROPOUT = 0.1
LSTM_RECURRENT_DROPOUT = 0.1

class ChitEncoder(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
            max_length: int,
    ) -> None:
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=EMBEDDING_DIMENTION,
            name='embedding',
        )
        self.lstm = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            name='encoder_lstm'
        )

    def call(self, inputs):
        output = self.embedding(inputs)
        output, state = self.lstm(output)
        return output, state

    def predict(self, inputs):
        pass


class ChitDecoder(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(
            units=LSTM_UNIT_NUM,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_state=True,
            return_sequences=True,
        )

        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=vocabulary_size,
                activation='softmax',
            )
        )

    def call(self, inputs):
        context, state = self.lstm(inputs)
        context = self.dense(context)
        output = self.softmax(context)
        return output, state

    def predict(self, inputs):
        pass


class ChitChat(tf.keras.Model):
    '''doc'''
    def __init__(
            self,
            vocabulary_size: int,
    ) -> None:
        super().__init__()

        self.encoder = ChitEncoder(
            vocabulary_size=vocabulary_size,
        )
        self.decoder = ChitDecoder()

    def call(self, inputs):
        batch_size, seq_length = tf.shape(inputs)
        inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        state = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        for t in range(seq_length.numpy()):
            output, state = self.cell(inputs[:, t, :], state)
        output = self.dense(output)
        return output

    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size.numpy())])


####################
learning_rate = 1e-3
num_batches = 10
batch_size = 128

data_loader = DataLoader()
vocabulary = Vocabulary(data_loader.raw_text)

model = ChitChat(
    vocabulary_size=data_loader.size+1,
    embedding_dimention=EMBEDDING_DIMENTION,
)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for batch_index in range(num_batches):
    sentence, next_sentence = data_loader.get_batch(batch_size)

    sequence = vocabulary.sentence_to_seqence(sentence)
    next_sequence = vocabulary.sentence_to_seqence(next_sentence)

    with tf.GradientTape() as tape:
        word_logit_pred = model(sequence)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=word_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


##################

X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print("diversity %f:" % diversity)
    for t in range(400):
        y_pred = model.predict(X, diversity)
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
