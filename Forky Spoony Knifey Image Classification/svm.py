from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import *

def train_svm_model(x_train, x_test, y_train, y_test,
    num_classes, grid_search_params, cpus):

    # reshape as 1-d arrays
    y_train = y_train.reshape((-1))
    y_test = y_test.reshape((-1))

    encoder = LabelEncoder()
    encoder.fit(y_train)
    enc_y_train = encoder.transform(y_train)
    enc_y_test = encoder.transform(y_test)

    svc = SVC(verbose=True)
    clf = GridSearchCV(svc, grid_search_params, cv=5, n_jobs=cpus, verbose=20)
    clf.fit(x_train, enc_y_train)

    try:
        y_pred = clf.predict(x_test)
        report = accuracy_score(enc_y_test, y_pred, normalize=True)
        print(f'Accuracy on the test set - \n\{clf}:')
    except:
        print('failed but continuing')

    return clf

def main():
    x_train, x_test, y_train, y_test = load_dataset()

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    x_train = x_train.reshape((train_size, -1))
    x_test = x_test.reshape((test_size, -1))

    classes = np.unique(y_train).shape[0]

    grid_search_params = [
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    ]
    clf = train_svm_model(x_train, x_test, y_train, y_test, classes, grid_search_params, cpus=3)

    save_object_to_disk('models/svc.obj', clf)

if __name__ == "__main__":
    main()