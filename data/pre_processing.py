# In this file, Divide the raw data into train, validation and test sets
# Make the index for the images and captions

import os
import shutil
import random
import string
import re

class PreProcessing:
    """
    This class is used to preprocess the data. 
    It is used to get the image list, the caption list, and to make the caption dictionary.
    """
    @staticmethod
    def get_image_list(path):
        """
        Returns the list of images in the given path.
        """
        return os.listdir(path)
    
    @staticmethod
    def get_caption_list(path):
        """
        Returns the list of captions in the given path.
        """
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            return lines
        
    @staticmethod
    def make_caption_dict(caption_list):
        """
        Returns the caption dictionary.
        Each key is an image name and the value is a list of captions for that image.
        """
        caption_dict = {}
        for line in caption_list:
            if ".jpg" in line:
                try:
                    name, caption = line.split(".jpg")
                    name = name + ".jpg"
                    caption = PreProcessing.clean_text(caption)
                    if name not in caption_dict:
                        caption_dict[name] = [caption]
                    else:
                        caption_dict[name].append(caption)
                except:
                    print(line)
        return caption_dict

    @staticmethod
    def clean_text(text: str):
        """
        Cleans the caption by removing \n, lowercasing, removing punctuation, and removing numbers.
        """
        text = text.replace("\n", "")
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text   
        
    @staticmethod
    def divide_images(image_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Divides the image list into train, validation, and test sets.
        """
        # shuffle the image list
        total_images = len(image_list)
        random.shuffle(image_list)
        
        train_images = image_list[:int(total_images * train_ratio)]
        val_images = image_list[int(total_images * train_ratio):int(total_images * (train_ratio+val_ratio))]
        test_images = image_list[int(total_images * (train_ratio+val_ratio)):]
        
        return train_images, val_images, test_images
        
    @staticmethod
    def make_index(image_list, caption_dict):
        """
        Makes the index for the images and captions.
        """
        counter = 0
        text_output = []
        for image in image_list:
            for caption in caption_dict[image]:
                text_output.append((counter, image, caption))
                counter += 1
        return text_output
        
    @staticmethod
    def save_data(image_list, caption_index, folder_name):
        """
        Saves the data to the given folder.
        """
        base_path = "data/raw/images"
        main_path = "data/processed/"
        os.makedirs(os.path.join(main_path, folder_name), exist_ok=True)
        os.makedirs(os.path.join(main_path, folder_name, "images"), exist_ok=True)
        
        for image in image_list:
            shutil.copy(os.path.join(base_path, image), os.path.join(main_path, folder_name, "images", image))
        
        with open(os.path.join(main_path, folder_name, "index.txt"), "w") as file:
            for index in caption_index:
                file.write(f"{index[0]},{index[1]},{index[2]}\n")


if __name__ == "__main__":
    print("Starting the pre-processing")
    pre_processing = PreProcessing()
    image_list = pre_processing.get_image_list("data/raw/images")
    print("Getting the caption list")
    print("Number of images: ", len(image_list))
    caption_list = pre_processing.get_caption_list("data/raw/captions.txt")
    print("Number of captions: ", len(caption_list))
    caption_dict = pre_processing.make_caption_dict(caption_list)
    
    print("Dividing the images into train, validation, and test sets")
    train_images, val_images, test_images = pre_processing.divide_images(image_list)
    
    train_index = pre_processing.make_index(train_images, caption_dict)
    val_index = pre_processing.make_index(val_images, caption_dict)
    test_index = pre_processing.make_index(test_images, caption_dict)
    
    print("Saving the data")
    pre_processing.save_data(train_images, train_index, "train")
    pre_processing.save_data(val_images, val_index, "val")
    pre_processing.save_data(test_images, test_index, "test")
    
    print("Done")
