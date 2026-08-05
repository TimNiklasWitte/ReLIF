import sys
sys.path.append("./..")


import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

from Classifier_ReLIF import *
from LoadDataframe import *

n_hidden = 128

def pi_formatter(x, pos):

    if x == 0:
        return '0'
    elif x == np.pi:
        return r'$\pi$'
    elif x == 2 * np.pi:
        return r'$2\pi$'
    else:
        # Calculate the fraction of π
        fraction = x / np.pi
        if abs(fraction - round(fraction)) < 0.01:
            return f'{int(round(fraction))}$\pi$'
        elif abs(fraction - round(fraction, 2)) < 0.01:
            return f'{fraction:.1f}$\pi$'
        else:
            return f'{x:.1f}'


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = "../logs"

    dataset_name_list = ["NMMNIST", "SHD"]
    freq_max_list = [1, 2, 4, 8]


    for dataset_name in dataset_name_list:
        
        if dataset_name == "NMMNIST":

            input_size = 68*68
            n_classes = 10
        
            

        elif dataset_name == "SHD":

            input_size = 700
            n_classes = 20

        

        for freq_max in freq_max_list:
            
            log_dir = f"{root}/{dataset_name}_ReLIF_{freq_max}"

            df = load_dataframe(log_dir)

            test_accuracy = df.loc[:, "test accuracy"]
            best_model_idx = np.argmax(test_accuracy)

            for epoch in [0, best_model_idx]:
                
    
                model = Classifier_ReLIF(input_size=input_size, n_hidden=n_hidden, n_classes=n_classes, freq_max=freq_max)

                checkpoint = torch.load(f"./../saved_models/{dataset_name}_ReLIF_{freq_max}/{epoch}", map_location=device)
                model.load_state_dict(checkpoint)

                
                fig = plt.figure(figsize=(14, 6))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  
         
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                cbar_ax = fig.add_subplot(gs[2])

            
                relif_1_freq = model.relif_1.freq_max * torch.sigmoid(model.relif_1.freq_logit.detach())
                relif_1_freq = relif_1_freq.numpy()

                relif_1_phase = 2 * torch.pi * torch.sigmoid(model.relif_1.phase_logit.detach())
                relif_1_phase = relif_1_phase.numpy()

                relif_1_resonating = torch.sigmoid(model.relif_1.resonating_logit.detach())
                relif_1_resonating = relif_1_resonating.numpy()

                scatter1 = ax1.scatter(relif_1_freq, relif_1_phase, c=relif_1_resonating, cmap='viridis', s=50, alpha=0.7, vmin=0, vmax=1)
                ax1.set_xlim(0, model.relif_1.freq_max)
                ax1.set_ylim(0, 2 * torch.pi)
                ax1.set_xlabel('Frequency', fontsize=12)
                ax1.set_ylabel('Phase', fontsize=12)
                ax1.set_title('Hidden Layer', fontsize=14)
                ax1.grid(True, alpha=0.3)

            
                ax1.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
                ax1.set_yticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', 
                                    r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'])

              
                relif_2_freq = model.relif_2.freq_max * torch.sigmoid(model.relif_2.freq_logit.detach())
                relif_2_freq = relif_2_freq.numpy()

                relif_2_phase = 2 * torch.pi * torch.sigmoid(model.relif_2.phase_logit.detach())
                relif_2_phase = relif_2_phase.numpy()

                relif_2_resonating = torch.sigmoid(model.relif_2.resonating_logit.detach())
                relif_2_resonating = relif_2_resonating.numpy()

                scatter2 = ax2.scatter(relif_2_freq, relif_2_phase, c=relif_2_resonating, cmap='viridis', s=50, alpha=0.7, vmin=0, vmax=1)
                ax2.set_xlim(0, model.relif_2.freq_max)
                ax2.set_ylim(0, 2 * torch.pi)
                ax2.set_xlabel('Frequency', fontsize=12)
                ax2.set_ylabel('Phase', fontsize=12)
                ax2.set_title('Output Layer', fontsize=14)
                ax2.grid(True, alpha=0.3)

            
                ax2.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
                ax2.set_yticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', 
                                    r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'])

               
                cbar = fig.colorbar(scatter1, cax=cbar_ax)
               
                scatter1.set_clim(0, 1)
                cbar.set_label('Resonating', fontsize=12)

                plt.suptitle(f"Epoch: {epoch}", fontsize=16)
                plt.tight_layout()
                os.makedirs(f"./plots/FreqPhasePlots/{dataset_name}_ReLIF_{freq_max}", exist_ok=True)
                plt.savefig(f"./plots/FreqPhasePlots/{dataset_name}_ReLIF_{freq_max}/Epoch_{epoch}.png", dpi=200)
                                
                plt.clf()
                plt.close()
                
        
         
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")