"""
Contains all the constants specific to the project.

To change model name, modify MODEL_NAME
"""

# Labels
CLASS_LABELS = ["Apple", "Banana", "Background"]    
LABEL_DICT = {"Apple": 0, "Banana": 1, "Background": 2}         # marks class names to corresponding classification numbers


# DATASET (Used in pipeline.py & util_fruits.py)
# util_fruits.py load data from PROCESSED_DATASET_PATH
RAW_DATASET_PATH = "C:\\Users\\Jack\\Desktop\\fruits_dataset"
PROCESSED_DATASET_PATH = "C:\\Users\\Jack\\Desktop\\fruits_dataset_32"      ### MODIFY
LABELS_PATH = PROCESSED_DATASET_PATH + "\\labels.txt"                      

# DATA COLLECTION: VIDEOS
VIDEO_CLASS = "Apple\\"            
VIDEO_PATH = "C:\\Users\\Jack\\Desktop\\Fruits_Videos\\" + VIDEO_CLASS
SAVED_IMAGES_PATH = "C:\\Users\\Jack\\Desktop\\Fruits_Videos\\Frame_images"


# MODEL
MODEL_EXT = ".h5"
MODEL_NAME = "fruits_classifier_32"          ### MODIFY THIS 


# Image input size to RESIZE to (used in pipeline.py)
input_shape = (32, 32)                            ### MODIFY THIS
