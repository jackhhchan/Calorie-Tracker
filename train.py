import argparse
import os
import winsound
import time
import keras
from contextlib import redirect_stdout

from util_fruit import get_fruit_data
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model

import keras.backend as K
K.set_learning_phase(1)

from CNN import CNN
import util_fruit
import constants

# Model
MODEL_EXT = constants.MODEL_EXT
MODEL_NAME = constants.MODEL_NAME


def main(args):
    """ Read data, perform data augmentation then compile, train and save the model. """
    # read data from disk
    X_train, X_val, X_test, Y_train, Y_val, Y_test = util_fruit.get_fruit_data()

    # build model
    model = get_model(model_name="fruit_classifier")
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # augment training and testing data
    datagen, X_train, X_val = augment_data(X_train, X_val)

    # save model and training history
    model_folder_path = util_fruit.make_model_folder(model_name=MODEL_NAME)

    # Callbacks
    callbacks=[]
    cp_callback = ModelCheckpoint("{}/{}_callback.hdf5".format(model_folder_path, MODEL_NAME),
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    period=10)
    callbacks.append(cp_callback)

    # train model and save train history    
    batch_size = args.batch_size
    epochs = args.epochs
    history = train(model, X_train, X_val, Y_train, Y_val, callbacks, datagen, batch_size=batch_size, epochs=epochs)

    # winsound.Beep(5000, 10)     # play sound after training completes
    
    
    save_model(model=model, save_folder_path=model_folder_path, model_name=MODEL_NAME, model_ext=MODEL_EXT)
    save_history(history=history, save_folder_path=model_folder_path)

    # evaluate model and save results-
    evaluate(model=model, datagen=datagen, X_test=X_test, Y_test=Y_test, batch_size=batch_size, save_folder_path=model_folder_path)


"""
Helper functions
"""
def get_model(model_name):
    """ 
    Return the convolutional neural network model.
    """
    model = CNN().get_model(model_name=model_name)

    return model

def train(model, X_train, X_val, Y_train, Y_val, callbacks, datagen, batch_size, epochs):
    """ Train the model with augmentation and return training history """
    
    print("[INFO] Training model...")
    print("[INFO[ Training data shape: {0}".format(X_train.shape))

    before = time.time()

    history = model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train)/batch_size,        # entire training set is used per epoch
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=callbacks)

    after = time.time()
    time_elapsed = after - before / 60  # return time elapsed in minutes
    print("[INFO] Training time elapsed: {:.2f}mins".format(time_elapsed))

    return history

def augment_data(X_train, X_val):
    """ Perform data augmentation on X_train and X_test """
    # Initiate Data Augmentation
    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    # Augment training data.

    datagen.fit(X_train)
    datagen.fit(X_val)

    return datagen, X_train, X_val


def save_history(history, save_folder_path=None, params=['acc', 'val_acc']):
    """
    Plot and show training history.
    (Optional) Save training history as image.

    Args:
    history -- kera's history object (returned from fit_generator)
    params -- measuring parameters
    """
    
    print("[INFO] Showing train and test accuracy plot...")

    # Plot all lines in parameters
    for param in params:
        plt.plot(history.history[param])
    
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    if save_folder_path is not None:
        save_path = "{}/{}".format(save_folder_path, "train_history.png")
        history_fig = plt.gcf()     # get current figure
        history_fig.savefig(save_path)
        print("[INFO] Plot saved to {0}".format(save_path))

    plt.show()

def save_model(model, save_folder_path, model_name, model_ext):
    """ Save model (including weights) to .h5 file and its summary in a .txt file """

    
    assert model_ext in [".h5", ".hdf5"]
    assert os.path.isdir(save_folder_path)==True, "model_folder_path is not a folder."

    print("[INFO] Saving model to {0}/".format(save_folder_path))
    model_path = "{}/{}".format(save_folder_path, model_name + model_ext)
    model.save(model_path)

    with open('{}/{}'.format(save_folder_path, "model_summary.txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    # visualize model architecture
    plot_model(model, to_file="{}/{}.png".format(save_folder_path, model_name),
                show_shapes=True)

def evaluate(model, datagen, X_test, Y_test, batch_size, save_folder_path=None):
    """ Evaluate the model with testing data """

    print("[INFO] Evaluating model...")

    scores = model.evaluate_generator(
        datagen.flow(X_test, Y_test, batch_size=batch_size),
        verbose=1)
    
    print("[INFO] Evaluation results:\n{0}: {1:.2f}\n{2}: {3:.2f}".format(model.metrics_names[0], scores[0]*100, model.metrics_names[1], scores[1]*100))
    
    if save_folder_path is not None:
        # Write results to path
        assert os.path.isdir(save_folder_path) == True, "Unable to save evaluation results, save_folder_path is not a folder"
        eval_results_path = save_folder_path + "/eval_results.txt"
        eval_handle = open(eval_results_path, 'w')
        eval_handle.write("Model name: {}\n\n".format(MODEL_NAME))
        eval_handle.write("Evaluation results:\n{0}: {1:.2f}\n{2}: {3:.2f}".format(model.metrics_names[0], scores[0]*100, model.metrics_names[1], scores[1]*100))
        eval_handle.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--batch_size',
        help='The batch size to use for training',
        required=False, type=int
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for",
        required=False, type=int
    )
    parser.set_defaults(epochs=12)
    parser.set_defaults(batch_size=8)

    args = parser.parse_args()    # insert in array to override defaults e.g. ['-e', '100', '-b', '128']

    main(args)