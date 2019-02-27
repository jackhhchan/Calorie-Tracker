"""
Note:   coreml uses python2.7
        Do not remove from folder directory.


convert_to_coreml

Converts .h5 model files trained using Keras to mlmodels. 
The mlmodel is saved in the same directory as the keras .h5 model.
"""

import coremltools
import keras
import sys
from keras.applications import MobileNet
from keras.models import load_model


import constants

MODEL_NAME = "fruits_classifier_256_deeper_by_2"        # MODIFY THIS


MODELS_FOLDER_PATH = "/Users/jackchan/OneDrive/Documents/Unimelb - MIT/CS Project Unimelb/Source Code/CNN_Training/Code (Final)/model"

MODEL_FOLDER_NAME = MODEL_NAME
MODEL_PATH = "{}/{}/{}.h5".format(MODELS_FOLDER_PATH, MODEL_FOLDER_NAME, MODEL_NAME)

MLMODEL_NAME = MODEL_NAME  # saved mlmodel name; .mlmodel extension is added in the program.


def main():
    """ Takes in keras model and convert to .mlmodel"""
    print(sys.version)

    # Load in keras model.
    model = load_model(MODEL_PATH)

    labels = constants.CLASS_LABELS
    print("[INFO] Labels to bind to mlmodel: [{0}]".format(labels))


    # Convert to .mlmodel
    coreml_model = coremltools.converters.keras.convert(
        model=model,
        input_names="image",
        image_input_names="image",
        is_bgr=True,
        image_scale=1/255.0,        # mlmodel loads 255 rgb values on default. (our models are trained with range 0.0-1.0)
        class_labels=labels)
    
    # Save .mlmodel
    print("[INFO] Saving coreml model in {}/{}/{}.mlmodel".format(MODELS_FOLDER_PATH, MODEL_FOLDER_NAME, MLMODEL_NAME))
    coreml_model.save("{}/{}/{}.mlmodel".format(MODELS_FOLDER_PATH, MODEL_FOLDER_NAME, MLMODEL_NAME))
    print("[INFO] Saved")


if __name__ == "__main__":
    main()
    # caffe()
