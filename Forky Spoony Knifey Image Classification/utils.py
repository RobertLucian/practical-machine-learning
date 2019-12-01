import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

def load_image(filename, width=None, height=None):
    img = Image.open(filename)
    img.load()
    if width and height:
        img = img.resize((width, height))
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
                    image = load_image(image_path, width=140, height=140)
                    dataset['x_' + test_or_train] += [image]
                    dataset['y_' + test_or_train] += [label]
    

    x = np.array(dataset['x_train'] + dataset['x_test']) / 255.0
    y = np.array(dataset['y_train'] + dataset['y_test']).reshape((-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

    return x_train, x_test, y_train, y_test

def save_trained_keras_model(model_path, model):
    """
    Save a trained model to disk.
    :param model_path: A relative/absolute path to the output model. Must not
    include and extension because there are 2 files written to disk: a json and an h5.
    I.e: "output/generic_model".
    :param model: The actual trained keras.models.Model object.
    :return: Nothing.
    """
    json = model.to_json(indent=2)
    with open(model_path + '.json', 'w') as json_file:
        json_file.write(json)
    model.save_weights(model_path + '.h5')

def load_pretrained_keras_model(model_path):
    """
    Load a pre-trained model.
    :param model_path: Relative path to an in-built resource of the package
    or an absolute path. Both the model and the weights must have the same name
    i.e "generic_model.json" and "generic_model.h5". When passing the argument,
    don't add the extension of it, because we're referring to two related files.
    :return: A keras.models.Model model.
    """

    with open(model_path + '.json', 'r') as f:
        loaded_json = f.read()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights(model_path + '.h5')

    return loaded_model

def save_object_to_disk(filename, obj):
    """
    Write an object to a file with pickle.
    :param filename: The pathname of the written file.
    :obj: The object that gets pickled. 
    """
    with open(filename, 'wb') as binf:
        pickle.dump(obj, binf)


def load_object_from_disk(filename):
    """
    Read an object from a file with pickle.
    :param filename: The pathname of the written file.
    """
    obj = None
    with open(filename, 'rb') as binf:
        obj = pickle.load(binf)
    
    return obj