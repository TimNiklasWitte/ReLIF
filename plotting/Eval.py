from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

def main():

    root = "../logs"

    dataset_name_list = ["NMMNIST", "SHD"]
    model_name_list = ["LIF", "ReLIF_1", "ReLIF_2", "ReLIF_4", "ReLIF_8"]


    for dataset_name in dataset_name_list:
        print(dataset_name)
        for model_name in model_name_list:

            log_dir = f"{root}/{dataset_name}_{model_name}"

            df = load_dataframe(log_dir)

            test_accuracy = df.loc[:, "test accuracy"]
            best_test_accuracy = np.max(test_accuracy)

            print(f"{model_name}: {best_test_accuracy:.5f}")
         
        
        print()
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")