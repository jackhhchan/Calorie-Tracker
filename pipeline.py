import os
import cv2
from tqdm import tqdm
import random
import argparse

"""
class Pipeline

contains methods for: Image renaming, data collection, image processing

Typical usage:
1. call rename_images() first, so that all images from the raw dataset are properly named as they are the same names used in the processed dataset
2. call process_images() to centre crop and resize images to the processed dataset
3. call image_to_labels() to generate a labels.txt in processed dataset
4. run train_models.py (separate program) --- make sure neural network input shape and resized images shape in processed dataset are the same.
"""

class Pipeline:

    """
    IMAGE RENAMING

    rename_images()         -- properly rename all images in all folders in the dataset path
    reset_images_names()    -- to be called before rename_images(), reset all images name so they can be properly renamed.
    """

    def rename_images(self, dataset_path, image_ext):
        """
        Rename all the images in the folders of dataset_path to format CLASS_num.jpg
        
        - CLASS is obtained from folder name.
        - num is iterated by 1 per image.
        """

        assert image_ext in ['.jpg', '.png']        

        self.reset_images_names(dataset_path=dataset_path)

        print("[INFO] Renaming images in {}...".format(dataset_path))
        # Loop over folders
        folders = os.listdir(dataset_path)
        for folder in folders:                                 
            folder_path = dataset_path + "\\" + folder          # construct absolute paths for folders
            if not os.path.isdir(folder_path): continue         # skip if path is not a folder
            
            # Loop over images in folders
            images = os.listdir(folder_path)
            iterate_num = 0                                                                         # begin at 0 for every folder
            for image in images:
                image_path = folder_path + "\\" + image                                             # construct absolute image path
                renamed_image = folder_path + "\\" + folder + "_" + str(iterate_num) + image_ext    # construct image name
                os.rename(image_path, renamed_image)                                                # rename image in image path
                iterate_num += 1

    def reset_images_names(self, dataset_path):
        """
        Reset all names in the images to avoid 'name already existing' during renaming.

        To be called before rename_images().
        """
        random_extensions = ['a', 'b', 'c', 'd', 'e']

        print("[INFO] Resetting all images name in {}...".format(dataset_path))
        folders = os.listdir(dataset_path)
        for folder in folders:
            folder_path = dataset_path + "\\" + folder
            if not os.path.isdir(folder_path): continue
            
            images = os.listdir(folder_path)
            iterate_num = 0                                 # reset iterate_num for every class
            for image in images:
                image_path = "{}/{}".format(folder_path, image)
                try:
                    # Rename image
                    renamed_image = folder_path + "\\" + str(iterate_num)
                    os.rename(image_path, renamed_image)
                except:
                    # append random extension to name if name already exists
                    random_index = random.randint(0, len(random_extensions)-1)
                    renamed_image = folder_path + "\\" + str(iterate_num) + random_extensions[random_index]
                    os.rename(image_path, renamed_image)
                
                iterate_num += 1


    """
    DATA COLLECTION  

    frame_to_images()       -- converts frames from all videos in folder to images
    """

    def frame_to_images(self, video_folder_path, saved_images_path, image_ext, per_frame=3, skip=[]):
        """ 
        Save video frame (every 'per_frame') from videos in video_folder_path to saved_images_path.
        """
        assert image_ext in [".jpg", '.png']
        print("[INFO] {} frames per image, image extension: {}")
        print("[INFO] Images will be saved to {}\n".format(saved_images_path))
        print("[INFO] Loading video from path: {0}...".format(video_folder_path))
        file_num = 0
        videos = os.listdir(video_folder_path)                                      # get all videos from folder path
        for video in tqdm(videos):
            if video in skip and len(skip) != 0:                                    # skip video to process if in skipped_video array        
                print("Skipped {0}".format(video))
                continue
            
            video_path = video_folder_path + "\\" + video
            print("[INFO] Reading from path: {0}...".format(video_path))

            cap = cv2.VideoCapture(video_path)                                      # load video from path to cap
            if not cap.isOpened():
                print("Error in opening {0}".format(video_path))
                return

            print("[INFO] Writing video frames to images...")
            frame_tracker = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if frame is None:              # exit if frame is none
                    break

                if frame_tracker % per_frame == 0:     # save every 3 frame
                    file_name = saved_images_path + "\\video_frame_" + str(file_num) + image_ext  # save frames as images to saved_images_path with file_num
                    cv2.imwrite(file_name, frame)
                    file_num += 1
                
                frame_tracker += 1

            cap.release()
            print("[INFO] Complete")


    """
    IMAGE PROCESSING
    ~processed dataset path

    All processed images are saved to /fruits_dataset_processed
    preprocess_images()    -- preprocess(center crop & resize) all images from original dataset to new folder
    crop_center()          -- center crop image dynamically based on height
    resize()               -- resize image to desired shape
    """

    def preprocess_images(self, input_shape, dataset_path, processed_dataset_path):
        """ Preprocess images (centre crop & resize) and save to processed_dataset_path. """

        print("[INFO] Preprocessing all images in {} to {}...".format(dataset_path, processed_dataset_path))
        print("[INFO] Resized image shape: ({0}, {1})".format(str(input_shape[0]), str(input_shape[1])))

        folders = os.listdir(dataset_path)
        for folder in tqdm(folders):                            
            folder_path = dataset_path + "/" + folder
            if not os.path.isdir(folder_path): continue         # skip if path is not a folder.
                
            images = os.listdir(folder_path)
            for image in tqdm(images):
                image_path = dataset_path + "/" + folder + "/" + image
                try:
                    image_to_process = cv2.imread(image_path)                    
                    cropped = self.crop_center(image=image_to_process)               # crop center image
                    resized = self.resize(image=cropped, resize_shape=input_shape)   # resize image to input shape
                except Exception as e:
                    print(e)
                    print("Image path at exception: {}".format(image_path))

                # set up image path in new folder
                assert os.path.isdir(processed_dataset_path), "Processed dataset path is not a directory."
                
                processed_folder_path = "{}/{}".format(processed_dataset_path, folder)
                if not os.path.isdir(processed_folder_path): os.mkdir(processed_folder_path)
                
                processed_image_path =  "{}/{}".format(processed_folder_path, image)
                cv2.imwrite(processed_image_path, resized)                       # write new image to processed dataset path
        
        print("[INFO] Complete")

    def crop_center(self, image):
        """ Dynamically crop the centre of the image based on image orientation. """
        image_height = image.shape[0]
        image_width = image.shape[1]
        
        square = image_height == image_width
        landscape = image_width > image_height

        if square:
            return image

        elif landscape:
            new_left_edge = int((image_width - image_height)/2)     # integer conversion may cause problem if not divisible by 2
            new_right_edge: int = image_width - new_left_edge
            image = image[:, new_left_edge: new_right_edge, :]
            # cv2.imshow("cropped image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return image

        elif not landscape:
            new_top_edge = int((image_height - image_width)/2)
            new_bottom_edge: int = (image_height) - new_top_edge
            image = image[new_top_edge: new_bottom_edge, :, :]
            # cv2.imshow("cropped image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return image

    def resize(self, image, resize_shape):
        """ Resize image using cv2.INTER_AREA"""
        assert len(resize_shape) == 2, "Resize shape can only be 2 arguments, e.g. (128, 128)"
        # convert resize_shape to tuple if not.
        if type(resize_shape) != tuple: resize_shape=tuple(resize_shape)

        image_resized = cv2.resize(image, resize_shape, interpolation=cv2.INTER_AREA)
        return image_resized


    """
    LABELLING
    ~processed dataset path

    images_to_labels()      -- make labels.txt with format (IMAGE_NAME    fruit_class_number) i.e IMAGE_NAME\tfruit_class_number\n
    """

    def images_to_labels(self, dataset_path, labels_path, label_dict):
        """
        Make labels text file from images in dataset folders to labels.txt

        FORMAT: labels.txt label format (e.g. Apple_0.jpg   0) i.e. Apple_0.jpg\t0\n

        """
        print("[INFO] Converting images to labels... ")

        label_handle = open(labels_path, 'w')
        folders = os.listdir(dataset_path)
        for folder in folders:
            folder_path = dataset_path + "/" + folder
            if not os.path.isdir(folder_path): continue                 # skip path if not folder

            images = os.listdir(folder_path)                             
            for image in images:
                fruit_class = image.split("_")[0]                                   # get fruit class from image file name
                fruit_class_num = label_dict.get(fruit_class)                       # get fruit class number
                label_handle.write(image + "\t" + str(fruit_class_num) + "\n")      # FORMAT: e.g. Apple_0.jpg\t0\n
        print("[INFO] Complete")





import constants
def main(args):
    pipeline = Pipeline()
    # pipeline.rename_images(args.raw_dataset, args.image_ext)
    pipeline.preprocess_images(args.input_shape, args.raw_dataset, args.processed_dataset)
    label_dict = constants.LABEL_DICT
    pipeline.images_to_labels(args.processed_dataset, args.label, label_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--raw_dataset', 
                        required=True, type=str, help="Input raw dataset path")
    parser.add_argument('-pd', '--processed_dataset', 
                        required=True, type=str, help="Output processed dataset path")
    parser.add_argument('-l', '--label', 
                        required=True, type=str, help="Output labels path.")
    parser.add_argument('-i', '--input_shape', nargs='+',
                        required=True, type=int, help="Image input shape e.g. (128, 128)")
    parser.add_argument('-ext', '--image_ext', 
                        required=False, type=str, help="Image extension to rename images")
    parser.set_defaults(image_ext='.jpg')
    
    args_to_parse = ['-rd', constants.RAW_DATASET_PATH, 
                    '-pd', constants.PROCESSED_DATASET_PATH,
                    '-l', constants.LABELS_PATH,
                    '-i', str(constants.input_shape[0]), str(constants.input_shape[0])]
    args = parser.parse_args(args_to_parse)
    main(args)