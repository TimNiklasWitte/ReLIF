# ReLIF: Resonator-extended Leaky Integrate-and-Fire Neuron


<img src="./plotting/plots/CompareIntRes.png" style="width:50%;">

# Training


<img src="./plotting/plots/AccuracyLoss_NMMNIST.png" style="width:50%;">

<img src="./plotting/plots/AccuracyLoss_SHD.png" style="width:50%;">

## Train accuracy

| Model | NMMNIST | SHD |
| :--- | :---: | :---: |
| LIF | 0.98656 | 0.92988 |
| ReLIF_1 | 0.99018 | **0.96335** |
| ReLIF_2 | **0.99085** | 0.95727 |
| ReLIF_4 | 0.99005 | 0.96200 |
| ReLIF_8 | 0.98956 | 0.96142 |

## Test accuracy

| Model | NMMNIST | SHD |
| :--- | :---: | :---: |
| LIF | **0.94531** | 0.72411 |
| ReLIF_1 | 0.93970 | **0.85625** |
| ReLIF_2 | 0.93960 | 0.82946 |
| ReLIF_4 | 0.93680 | 0.84152 |
| ReLIF_8 | 0.93800 | 0.85089 |

# Phase Frequency Plots

## Neuromorphic MNIST (NMMNIST)

### ReLIF_1

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_1/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_1/Epoch_98.png" style="width:75%;">

### ReLIF_2

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_2/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_2/Epoch_98.png" style="width:75%;">

### ReLIF_4

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_4/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_4/Epoch_93.png" style="width:75%;">

### ReLIF_8

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_8/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/NMMNIST_ReLIF_8/Epoch_75.png" style="width:75%;">

## Spiking Heidelberg Digits (SHD)

### ReLIF_1

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_1/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_1/Epoch_86.png" style="width:75%;">

### ReLIF_2

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_2/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_2/Epoch_69.png" style="width:75%;">

### ReLIF_4

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_4/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_4/Epoch_93.png" style="width:75%;">

### ReLIF_8

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_8/Epoch_0.png" style="width:75%;">

<img src="./plotting/plots/FreqPhasePlots/SHD_ReLIF_8/Epoch_83.png" style="width:75%;">