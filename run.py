from data.data_set import DataSet

if __name__ == "__main__":
    data_set = DataSet("data/raw/images", "data/raw/captions.txt")
    print(len(data_set))
