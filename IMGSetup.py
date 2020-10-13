import numpy as np
import os
import cv2
import random
import pickle

IMG_SIZE = 100
DATADIR = "data"
CATEGORIES = ["apple", "apricot", "avocado", "banana", "blackberry", "blueberry", "cherry", "coconut",
                  "fig", "grape", "grapefruit", "kiwifruit", "lemon", "lime", "mango", "olive", "orange",
                  "passionfruit", "peach", "pear", "pineapple", "plum", "pomegranate", "raspberry",
                  "strawberry", "tomato", "watermelon"]

def createData():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                trainingData.append([new_array, class_num])
            except Exception as e:
                pass

trainingData = []
createData()
random.shuffle(trainingData)

X = [] # Features
Y = [] # Labels

for features, label in trainingData:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = np.array(X)
Y = np.array(Y)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
