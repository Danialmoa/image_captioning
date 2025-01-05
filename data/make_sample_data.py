import os
import shutil
import random

def make_sample_data(sample_size):
    path = "data/processed/test/"
    with open(path + "index.txt", "r") as file:
        lines = file.readlines()
        
    random.shuffle(lines)
    os.makedirs("data/sample_data/test/", exist_ok=True)
    with open("data/sample_data/test/index.txt", "w") as file:
        for line in lines[:sample_size]:
            file.write(line)
    
    for line in lines[:sample_size]:
        image_name = line.split(",")[1]
        image_path = path + "images/" + image_name
        os.makedirs("data/sample_data/test/images/", exist_ok=True)
        shutil.copy(image_path, "data/sample_data/test/images/" + image_name)
    
    return lines


if __name__ == "__main__":
    number_of_images = 10
    make_sample_data(number_of_images)