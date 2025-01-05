import os
import shutil


def make_sample_data(sample_size):
    path = "data/processed/test/"
    with open(path + "index.txt", "r") as file:
        lines = file.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = [line for line in lines if int(line[0]) <= sample_size]
    with open("data/sample_data/index.txt", "w") as file:
        for line in lines:
            file.write(line[0] + "," + line[1] + "\n")
    
    for line in lines:
        image_name = line[1]
        image_path = path + "images/" + image_name
        os.makedirs("data/sample_data/images/", exist_ok=True)
        shutil.copy(image_path, "data/sample_data/images/" + image_name)
    
    return lines


if __name__ == "__main__":
    number_of_images = 10
    make_sample_data(number_of_images * 5)