import numpy as np
from PIL import Image

def load_japanese_mnist():
    folder_path = '/Users/romc/Documents/RNN_exploration_learning/LearningRate/input/japanese_MNIST/'

    with np.load(folder_path + 'kmnist-train-imgs.npz') as data:
        # Note that the keys are the names we created above (i.e. filename sans .tif suffix)
        X_train = data['arr_0']

    with np.load(folder_path + 'kmnist-train-labels.npz') as data:
        y_train = data['arr_0']
    return X_train, y_train

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
