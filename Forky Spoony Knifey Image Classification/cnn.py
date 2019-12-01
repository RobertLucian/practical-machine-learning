import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras.layers as layer
import tensorflow as tf

from utils import *

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def BatchNormalization(trainable=True):
    bn = layer.BatchNormalization()
    bn.trainable = True
    return bn

def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(layer.Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(layer.Activation('relu'))
    # model.add(layer.Dropout(0.2))
    model.add(layer.MaxPooling2D(pool_size=(3,3)))

    model.add(layer.Conv2D(filters=64, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(layer.Activation('relu'))
    # model.add(layer.Dropout(0.2))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layer.Conv2D(filters=128, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(layer.Activation('relu'))
    # model.add(layer.Dropout(0.2))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layer.Conv2D(filters=128, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(layer.Activation('relu'))
    # model.add(layer.Dropout(0.3))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))

    model.add(layer.Flatten())
    model.add(layer.Dense(92, activation='relu'))
    model.add(layer.Dropout(0.3))
    model.add(layer.Dense(92, activation='relu'))
    model.add(layer.Dropout(0.3))
    model.add(layer.Dense(92, activation='relu'))
    model.add(layer.Dropout(0.3))
    model.add(layer.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_cnn_model(x_train, x_test, y_train, y_test,
    k, num_classes, cpus):
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    test_scores = []
    train_hists = []
    input_shape = x_train.shape[1:]

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(y_train)
    enc_y_train = encoder.transform(y_train).toarray()
    enc_y_test = encoder.transform(y_test).toarray()

    model = build_model(input_shape, num_classes)
    for train, test in kfold.split(x_train, y_train):

        history = model.fit(x_train[train], enc_y_train[train], epochs=45, batch_size=64, validation_data=(x_train[test], enc_y_train[test]), use_multiprocessing=True, workers=cpus)
        scores = model.evaluate(x_test, enc_y_test)

        print(f'{model.metrics_names[1]}: {scores[1] * 100}')
        test_scores.append(scores)
        train_hists.append(history)

    return train_hists, test_scores, model

def main():
    x_train, x_test, y_train, y_test = load_dataset()
    classes = np.unique(y_train).shape[0]

    train_hists, test_scores, model = train_cnn_model(
        x_train, x_test, y_train, y_test, 
        k=10, num_classes=classes, cpus=6)

    save_object_to_disk('stats/cnn_train_hists.obj', train_hists)
    save_object_to_disk('stats/cnn_test_scores.obj', test_scores)
    save_trained_keras_model('models/cnn', model)

if __name__ == "__main__":
    main()