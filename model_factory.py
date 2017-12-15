import time

import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.layers import Dense, Dropout, Bidirectional, GRU, MaxPooling1D, Conv1D, Masking, Flatten
from keras.preprocessing import sequence


def get_model(model_name):
    if model_name in globals():
        return globals()[model_name]
    else:
        raise ValueError("Model '{}' doesn't exist!".format(model_name))


# ancillary methods
def one_hot_encode(number, size_vector):
    if number >= size_vector:
        raise ValueError('number must be less than size_vector')
    encoded = np.zeros(size_vector, dtype='float32')
    # This way allows us to get zeros vectors for padding
    if number >= 0:
        encoded[number] = 1.0

    return encoded


def fit_model(model_option, train, w2v_model):
    # line -> [tokens, deptree, conditions, candidates]
    # candidate -> [token_indexes, tokens_and_deptag, score]
    x = [candidate[1] for line in train for candidate in line[3]]
    y = [candidate[2] for line in train for candidate in line[3]]

    start = time.time()
    model = model_option(w2v_model)
    end = time.time()
    print('Model created', (end - start))
    print(model.summary())

    early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=1, mode='auto')

    start = time.time()
    model.fit(x, y, batch_size=32, epochs=150, verbose=2, callbacks=[early_stop_callback])
    end = time.time()
    print('Fit done! {}'.format((end - start)))

    return model


# Here we defined all our models (fit, predict, summary) -> wrappers methods
# fit -> input: list of candidates with scores
# predict -> input:
class ModelBase(object):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        self.wv = wv
        self.maxlen = maxlen
        self.max_num_deptag = max_num_deptag
        self.model = None

    def summary(self):
        self.model.summary()

    def compile_model(self):
        raise Exception('Not implemented!')

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        raise Exception('Not implemented!')

    def predict(self, x):
        raise Exception('Not implemented!')


class MLPLinear(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Dense(50, activation='tanh')(embedded_sequences)
        x = (Dropout(0.2))(x)
        x = Flatten()(x)
        preds = Dense(1)(x)
        self.model = Model(sequence_input, preds, name='MLPLinear')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class MLPSigmoid(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Dense(50, activation='tanh')(embedded_sequences)
        x = (Dropout(0.2))(x)
        x = Flatten()(x)
        preds = Dense(1, activation='sigmoid')(x)
        self.model = Model(sequence_input, preds, name='MLPSigmoid')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class MLPRelu(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Dense(50, activation='tanh')(embedded_sequences)
        x = (Dropout(0.2))(x)
        x = Flatten()(x)
        preds = Dense(1, activation='relu')(x)
        self.model = Model(sequence_input, preds, name='MLPRelu')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class MLPSoftmax(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Dense(50, activation='tanh')(embedded_sequences)
        x = (Dropout(0.2))(x)
        x = Flatten()(x)
        preds = Dense(1, activation='softmax')(x)
        self.model = Model(sequence_input, preds, name='MLPSoftmax')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array([[t[0] for t in row] for row in x])
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)