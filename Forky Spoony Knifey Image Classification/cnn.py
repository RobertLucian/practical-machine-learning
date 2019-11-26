import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
import keras.layers as layer
import tensorflow as tf

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype='float32')
    return data

def load_dataset():
    rootdir = os.path.dirname(os.path.abspath(__file__)) + '/dataset'
    dataset = {
        'x_train': [],
        'y_train': [],
        'x_test': [],
        'y_test': []
    }

    for subdir, _, files in os.walk(rootdir):
        if files:
            head, label = os.path.split(subdir)
            _, test_or_train =  os.path.split(head)
            test_or_train = 'test' if test_or_train == 'valid' else test_or_train
            if test_or_train in ['train', 'test']:
                print('Loading', test_or_train, 'data for', label)

        if len(files) > 1:
            for file in tqdm(files):
                _, ext = os.path.splitext(file)
                if ext == '.jpg':
                    image_path = os.path.join(subdir, file)
                    image = load_image(image_path)
                    dataset['x_' + test_or_train] += [image]
                    dataset['y_' + test_or_train] += [label]
    

    x_train = np.array(dataset['x_train']) / 255.0
    y_train = np.array(dataset['y_train']).reshape((-1,1))
    x_test = np.array(dataset['x_test']) / 255.0
    y_test = np.array(dataset['y_test']).reshape((-1,1))

    # labels = np.unique(y_train, return_inverse=True)[1]
    # y_train = np.vstack((labels, y_train)).T
    # labels = np.unique(y_test, return_inverse=True)[1]
    # y_test = np.vstack((labels, y_test)).T

    return x_train, y_train, x_test, y_test

def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(layer.Conv2D(filters=16, kernel_size=(5,5), input_shape=input_shape, data_format='channels_last', activation='relu'))
    model.add(layer.BatchNormalization())
    model.add(layer.Dropout(0.5))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))
    model.add(layer.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(layer.BatchNormalization())
    model.add(layer.Dropout(0.5))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))
    model.add(layer.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(layer.BatchNormalization())
    model.add(layer.Dropout(0.5))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))
    model.add(layer.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(layer.BatchNormalization())
    model.add(layer.Dropout(0.5))
    model.add(layer.MaxPooling2D(pool_size=(2,2)))
    model.add(layer.Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(layer.BatchNormalization())
    model.add(layer.Dropout(0.5))
    model.add(layer.MaxPooling2D(pool_size=(3,3)))

    model.add(layer.Flatten())
    model.add(layer.Dense(128, activation='relu'))
    model.add(layer.Dropout(0.3))
    model.add(layer.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_cnn_model(x_train, y_train, x_test, y_test,
    k, num_classes, cpus):
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    cv_scores = []
    input_shape = x_train.shape[1:]

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(y_train)
    enc_y_train = encoder.transform(y_train).toarray()
    enc_y_test = encoder.transform(y_test).toarray()

    # for train, test in kfold.split(x_train, y_train):
    #     model = build_model(input_shape, num_classes)
    #     model.fit(x_train[train], enc_y_train[train], epochs=50, batch_size=16, validation_data=(x_train[test], enc_y_train[test]), use_multiprocessing=True, workers=cpus)

    #     scores = model.evaluate(x_test, enc_y_test)
    #     print(f'{model.metrics_names[1]}: {scores[1] * 100}')
    #     cv_scores.append(scores)

    model = build_model(input_shape, num_classes)
    model.fit(x_train, enc_y_train, epochs=50, batch_size=32, validation_split=0.15, use_multiprocessing=True, workers=cpus)

    return cv_scores

def main():
    x_train, y_train, x_test, y_test = load_dataset()
    classes = np.unique(y_train).shape[0]
    scores = train_cnn_model(x_train, y_train, x_test, y_test, k=3, num_classes=classes, cpus=6)
    print(scores)

if __name__ == "__main__":
    main()