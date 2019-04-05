import numpy as np
from PIL import Image
from random import randint
import tensorflow as tf
import scipy.io as sio

def load_japanese_mnist():
    folder_path = '/Users/romc/Documents/RNN_exploration_learning/LearningRate/input/japanese_MNIST/'

    with np.load(folder_path + 'kmnist-train-imgs.npz') as data:
        # Note that the keys are the names we created above (i.e. filename sans .tif suffix)
        X_train = data['arr_0']

    with np.load(folder_path + 'kmnist-train-labels.npz') as data:
        y_train = data['arr_0']
    return X_train, y_train

def load_SVHN():
    data = sio.loadmat('/Users/romc/Documents/RNN_exploration_learning/LearningRate/input/SVHN/train_32x32.mat')
    X_train = data['X']
    y_train = data['y']
    y_train = np.squeeze(y_train, axis=-1)
    X_train = X_train.reshape(X_train.shape[-1],X_train.shape[0],X_train.shape[1],X_train.shape[2])
    return X_train, y_train


def sample_dataset():
    datasets = ['MNIST', 'FashionMNIST' , 'JapaneseMNIST', 'CIFAR10']
    num_datasets = len(datasets)
    # sample
    ind = randint(0, num_datasets-1)
    type = datasets[ind]
    if type == 'MNIST':
        (X_train, Y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.astype(float) / 255.
    elif type == 'FashionMNIST':
        (X_train, Y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.astype(float) / 255.
    elif type == 'CIFAR10':
        (X_train, Y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.astype(float) / 255.
        Y_train = np.squeeze(Y_train, axis=-1)
    elif type == 'JapaneseMNIST':
        X_train, Y_train = load_japanese_mnist()
        X_train = X_train.astype(float) / 255.
    return X_train, Y_train, type

'''
def load_natural_images():
    LABELS = {"airplane":0,  "car":1,  "cat":2,  "dog":3,  "flower":4,  "fruit":5,  "motorbike":6,  "person":7}
    LABELS_R = {}
    for itm in LABELS.keys():
        LABELS_R[LABELS[itm]] = itm
    folder_path = '/Users/romc/Documents/RNN_exploration_learning/LearningRate/input/natural_images/'

    divider=20
    objTrain=[]
    objTest=[]
    IMAGE_SIZE=(28,28)
    for c in os.listdir(folder_path):
        for m in os.listdir'/Users/romc/Documents/RNN_exploration_learning/LearningRate/input/natural_images/'+c):
            picpath='/Users/romc/Documents/RNN_exploration_learning/LearningRate/input/natural_images/'+c+"/"+m
            nodeInfo={"imgdir":picpath, "label":LABELS[c]}
            _magic = np.random.choice(range(divider))
            if _magic==0: objTest.append(nodeInfo)
            else: objTrain.append(nodeInfo)
    np.random.shuffle(objTrain)
    np.random.shuffle(objTest)

    X_train = []
    y_train = []

    for obj in object_train:
        infile = obj['imgdir']
        im = Image.open(infile)
        #im= im.resize(IMAGE_SIZE)
        #3. normalnize the data between 0,1
        np_im = np.array(im) / 255.0
        print(np_im.shape)



    return X_train, y_train
'''

if __name__ == "__main__":

    X_train, y_train = load_SVHN()

    print(X_train.shape)
    print(y_train.shape)
