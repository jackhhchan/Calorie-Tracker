import os
import cv2
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import constants

DATASET_PATH = constants.PROCESSED_DATASET_PATH                 
LABELS_PATH = constants.LABELS_PATH                            

label_dict = constants.LABEL_DICT

CLASS_LABELS = constants.CLASS_LABELS                          

dataset_split_ratio = (0.8, 0.05, 0.15)

def get_fruit_data(dataset_path=DATASET_PATH, labels_path=LABELS_PATH, onehot=True):
    """
    Load dataset and labels from disk, split the dataset, then return the train, validation & test data.
    
    Returns:
    X_train     -- array of training image data
    X_val       -- array of validation image data
    X_test      -- array of test image data
    Y_train     -- array of training labels
    Y_val       -- array of validation labels
    Y_test      -- array of test labels
    """


    # Load dataset and labels
    X = load_data(dataset_path=dataset_path)
    Y = load_labels(labels_path=labels_path)
    
    # Convert dataset to numpy arrays and normalize bgr values to within 0.0 - 1.0
    X, Y = convert_to_np_array(typeX='float32', typeY='uint8', X=X, Y=Y)
    X = normalize_data(X=X)
    
    # Split dataset into train and test.
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X=X, Y=Y, split=dataset_split_ratio)


    # Convert labels to one hot encoding.
    if onehot:
        print("[INFO] Converting labels to one hot encoding...")
        num_classes = len(CLASS_LABELS)
        Y_train, Y_val, Y_test = label_to_one_hot_encoding(Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                                            num_classes=num_classes)


    print(  "[INFO] Dataset summary:\n",
            "X_train: {0}\n".format(np.shape(X_train)),
            "X_val: {0}\n".format(np.shape(X_val)),
            "X_test: {0}\n".format(np.shape(X_test)), 
            "Y_train: {0}\n".format(np.shape(Y_train)),
            "Y_val: {0}\n".format(np.shape(Y_val)),
            "Y_test: {0}\n".format(np.shape(Y_test)))

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


""" Helper functions """

def load_data(dataset_path):
    """
    Load entire dataset from disk to memory in array X.

    args:
    dataset_path    -- absolute path to dataset containing images in folders corresponing to each class.
    return:
    X               -- array of images containing entire dataset
    """
    X = list()
    print("[INFO] Loading dataset into array from {0}...".format(dataset_path))
    folders = os.listdir(dataset_path)
    for folder in tqdm(folders):
        folder_path = dataset_path + "/" + folder
        if not os.path.isdir(folder_path): continue                  # skip if not folder

        for img_name in tqdm(os.listdir(folder_path)):
            img_path = folder_path + "/" + img_name
            img = cv2.imread(img_path, 1)                           # read image from path
            X.append(img)                              
    
    return X

def load_labels(labels_path):
    """    
    Load label class numbers into array Y.
    
    Note: labels.txt label format (e.g. Apple_0.jpg   0) i.e. Apple_0.jpg\t0
          Created from class Pipeline's images_to_labels() method
    """
    Y = list()          # initiate Y array to store labels

    print("[INFO] Loading labels into array from {0}...".format(labels_path))
    labels_handle = open(labels_path,'r')
    for line in labels_handle:
        line = line.split()
        Y.append(line[1])
    labels_handle.close()

    return Y

def convert_to_np_array(typeX: str, typeY: str, X, Y):
    """Convert loaded data into np arrays with types, typeX and typeY for data and labels correspondingly."""

    assert typeX == "float32"
    assert typeY == "uint8"

    X = np.array(X).astype(typeX)
    Y = np.array(Y).astype(typeY)


    print("[INFO] Dataset image shape: {0}".format(str(X[0].shape)))     # check image shape of first image

    return X, Y

def normalize_data(X):
    """
    Convert image data to range between 0.0 to 1.0 (float) instead of 0 to 255 (uint8, default RGB range).
    Reason:
    - native uint8 pixel values easily over/underfloat.
    - using float is more precise.
    """

    X = X/255.0

    return X

def label_to_one_hot_encoding(Y_train, Y_val, Y_test, num_classes):
    """ Convert labels to one hot encodings"""
    num_classes = np.int64(num_classes) 
    
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_val = np_utils.to_categorical(Y_val, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    return Y_train, Y_val, Y_test

def split_data(X, Y, split):
    """ Split the dataset to training, validation and test sets based on split ratio (train, val, test) """

    print("[INFO] Splitting datasets into train, validation and test sets...")

    train_portion = float(split[0])
    val_portion = float(split[1])
    test_portion = float(split[2])
    assert sum([train_portion, val_portion, test_portion]) == 1.0, "Combined split portions does not equal 100\%"

    test_size_1 = 1 - train_portion                             # get total test size for validation and test set
    test_size_2 = test_portion/(val_portion + test_portion)     # determine % of test set within validation + test

    # split to train, validation and test sets
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=test_size_1, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=test_size_2, random_state=42)

    # show class distribution of training data
    show_class_distribution(Y_train, data_for="train")
    show_class_distribution(Y_val, data_for="validation")
    show_class_distribution(Y_test, data_for="test")



    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def show_class_distribution(labels, data_for):
    """
    Print distribution of data between classes in one of the split dataset.
    labels -- not in one-hot encodings
    """
    assert data_for in ["train", "validation", "test"]

    num_0 = 0; num_1 = 0; num_2=0

    for label in labels:
        if label == label_dict.get("Apple"):
            num_0 += 1
        elif label == label_dict.get("Banana"):
            num_1 += 1
        elif label == label_dict.get("Background"):
            num_2 += 1
        else:
            print("not a class.")

    print("[INFO] Distribution of {} data".format(data_for),
        "Apple: {0:.2f}".format(num_0/len(labels)),
        "Banana: {0:.2f}".format(num_1/len(labels)),
        "Background: {0:.2f}".format(num_2/len(labels)))


def make_model_folder(model_name):
    """ make model folder in ~model/ and return its path """
    save_folder_path = "model/{}".format(model_name)
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)

    return save_folder_path



""" Sanity check functions """
def show_image_and_label(dataset_size, X, labels):
    """
    SANITY CHECK FUNCTION
    Show random image and its corresponding label before it is used for the neural network.

    Purpose:
    Used to check if they match.    
    """
    
    # Show random image (separate window) and corresponding label (in command prompt) from dataset
    for i in tqdm(range(dataset_size)):
        ran_number = random.randint(0 , len(X))
        print("Random Number: {0}".format(ran_number))

        print("Label: {0}".format(labels[ran_number]))
        cv2.imshow('X[{0}]'.format(ran_number), X[ran_number])
        cv2.waitKey(0)
        cv2.destroyAllWindows()





