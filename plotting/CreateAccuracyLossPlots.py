from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

def main():

    root = "../logs"

    dataset_name_list = ["NMMNIST", "SHD"]
    model_name_list = ["LIF", "ReLIF_1", "ReLIF_2", "ReLIF_4", "ReLIF_8"]

    for dataset_name in dataset_name_list:
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        df_list = []
        for model_name in model_name_list:
            log_dir = f"{root}/{dataset_name}_{model_name}"
            df = load_dataframe(log_dir)
            df["model name"] = model_name
            df_list.append(df)

        df = pd.concat(df_list)
        
        # Create lineplots with legend='full' to get proper labels
        sns.lineplot(data=df, x="Epoch", y="train accuracy", hue="model name", ax=axes[0][0], alpha=0.7, legend='full')
        axes[0][0].get_legend().remove()
        axes[0][0].set_title("Train accuracy")
        axes[0][0].set_ylabel("Accuracy")
        axes[0][0].grid(True)

        sns.lineplot(data=df, x="Epoch", y="test accuracy", hue="model name", ax=axes[1][0], alpha=0.7, legend='full')
        axes[1][0].get_legend().remove()
        axes[1][0].set_title("Test accuracy")
        axes[1][0].set_ylabel("Accuracy")
        axes[1][0].grid(True)

        sns.lineplot(data=df, x="Epoch", y="train loss", hue="model name", ax=axes[0][1], alpha=0.7, legend='full')
        axes[0][1].get_legend().remove()
        axes[0][1].set_title("Train loss")
        axes[0][1].set_ylabel("Loss")
        axes[0][1].grid(True)

        sns.lineplot(data=df, x="Epoch", y="test loss", hue="model name", ax=axes[1][1], alpha=0.7, legend='full')
        axes[1][1].get_legend().remove()
        axes[1][1].set_title("Test loss")
        axes[1][1].set_ylabel("Loss")
        axes[1][1].grid(True)

        # Get handles and labels from first subplot
        handles, labels = axes[0][0].get_legend_handles_labels()
        
        # Create single legend below the subplots (moved closer with bbox_to_anchor)
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), 
                ncol=3, frameon=True, fontsize=9)

        if dataset_name == "NMMNIST":
            axes[0][0].set_ylim(0.9, 1.0)
            axes[1][0].set_ylim(0.9, 0.95)
            axes[0][1].set_ylim(0.01, 0.2)
            axes[1][1].set_ylim(0.2, 0.5)
        
        elif dataset_name == "SHD":
            axes[0][0].set_ylim(0.80, 0.98)
            axes[1][0].set_ylim(0.6, 0.9)
            axes[0][1].set_ylim(0.1, 0.5)
            axes[1][1].set_ylim(0.5, 0.9)

        plt.suptitle(dataset_name, y=0.995)
        
        # Adjust bottom margin to reduce space (0.10 instead of 0.12)
        plt.tight_layout(rect=[0, 0.10, 1, 0.98])
        plt.savefig(f"./plots/AccuracyLoss_{dataset_name}.png", dpi=200, bbox_inches='tight')
        
        plt.clf()
        plt.close()
     
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")