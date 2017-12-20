import time

import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.layers import Dense, Dropout, Masking, Flatten, Conv1D, GlobalMaxPooling1D, GRU, Bidirectional, MaxPooling1D
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
    x = [candidate[1] for line in train for candidate in line[2]]
    y = [candidate[2] for line in train for candidate in line[2]]

    start = time.time()
    model = model_option(w2v_model)
    end = time.time()
    print('Model created', (end - start))
    print(model.summary())

    early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=1, mode='auto')

    start = time.time()
    model.fit(x, y, batch_size=32, epochs=1, verbose=2, callbacks=[early_stop_callback])
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
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
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
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
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
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
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
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class MLPSigmoid2(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Dense(50, activation='tanh')(embedded_sequences)
        x = (Dropout(0.2))(x)
        x = Flatten()(x)
        x = Dense(50, activation='tanh')(x)
        preds = Dense(1, activation='sigmoid')(x)
        self.model = Model(sequence_input, preds, name='MLPSigmoid2')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class MLPRelu2(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Dense(50, activation='tanh')(embedded_sequences)
        x = (Dropout(0.2))(x)
        x = Flatten()(x)
        x = Dense(50, activation='tanh')(x)
        preds = Dense(1, activation='relu')(x)
        self.model = Model(sequence_input, preds, name='MLPRelu2')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class CNNRelu(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (Conv1D(filters=128, kernel_size=3, activation='relu'))(embedded_sequences)
        # x = (AveragePooling1D(pool_size=3))(x)
        x = (Dropout(0.2))(x)
        x = (Conv1D(filters=32, kernel_size=17, activation='relu'))(x)
        x = (Dropout(0.2))(x)
        x = (GlobalMaxPooling1D())(x)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='relu')(x)
        self.model = Model(sequence_input, preds, name='CNNRelu')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class CNNSigmoid(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (Conv1D(filters=128, kernel_size=3, activation='relu'))(embedded_sequences)
        # x = (AveragePooling1D(pool_size=3))(x)
        x = (Dropout(0.2))(x)
        x = (Conv1D(filters=32, kernel_size=17, activation='relu'))(x)
        x = (Dropout(0.2))(x)
        x = (GlobalMaxPooling1D())(x)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='sigmoid')(x)
        self.model = Model(sequence_input, preds, name='CNNSigmoid')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class GRURelu(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (GRU(50, dropout=0.15))(embedded_sequences)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='relu')(x)
        self.model = Model(sequence_input, preds, name='GRURelu')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class GRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (GRU(50, dropout=0.15))(embedded_sequences)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='sigmoid')(x)
        self.model = Model(sequence_input, preds, name='GRUSigmoid')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class BiGRURelu(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (Bidirectional(GRU(50, dropout=0.15)))(embedded_sequences)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='relu')(x)
        self.model = Model(sequence_input, preds, name='BiGRURelu')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class BiGRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (Bidirectional(GRU(50, dropout=0.15)))(embedded_sequences)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='sigmoid')(x)
        self.model = Model(sequence_input, preds, name='GRUSigmoid')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class CNNBiGRURelu(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (Conv1D(filters=32, kernel_size=3, activation='relu'))(embedded_sequences)
        x = (MaxPooling1D(pool_size=2))(x)
        x = (Bidirectional(GRU(100, dropout=0.15)))(x)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='relu')(x)
        self.model = Model(sequence_input, preds, name='CNNBiGRURelu')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)


class CNNBiGRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=50, max_num_deptag=50):
        super().__init__(wv, maxlen, max_num_deptag)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = (Conv1D(filters=32, kernel_size=3, activation='relu'))(embedded_sequences)
        x = (MaxPooling1D(pool_size=2))(x)
        x = (Bidirectional(GRU(100, dropout=0.15)))(x)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = Dense(1, activation='sigmoid')(x)
        self.model = Model(sequence_input, preds, name='CNNGRUSigmoid')
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        y_preprocessed = np.array(y)

        self.model.fit(x_preprocessed, y_preprocessed, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       callbacks=callbacks)

    def predict(self, x):
        x_preprocessed = np.array(x)
        x_preprocessed = sequence.pad_sequences(x_preprocessed, maxlen=self.maxlen, value=-1)

        return self.model.predict(x_preprocessed)
