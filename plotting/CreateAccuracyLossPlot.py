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
            print(model_name, np.max(test_accuracy))
         
        
        print()
    # fig, axes = plt.subplots(1, 2)
    
    # sns.lineplot(data=df.loc[:, ["train accuracy", "test accuracy"]], ax=axes[0])#, markers=True)
    # axes[0].set_title("Accuracy")

    # sns.lineplot(data=df.loc[:, ["train loss", "test loss"]], ax=axes[1])#, markers=True)
    # axes[1].set_title("Loss")

    # # grid
    # for ax in axes.flatten():
    #     ax.grid()

    # plt.tight_layout()
    # #plt.savefig("./plots/AccuracyLoss.png", dpi=200)
    # plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")