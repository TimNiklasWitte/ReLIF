from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

def main():

    T = 300
    alpha = 0.9
    freq = 5

    V_lif = np.zeros(T, dtype=np.float32)
    V_res = np.zeros(T, dtype=np.float32)

    V_lif_plot = np.zeros(T, dtype=np.float32)
    V_res_plot = np.zeros(T, dtype=np.float32)


    spk_lif = np.zeros(T, dtype=np.float32)
    spk_res = np.zeros(T, dtype=np.float32)

    input_spikes_list = [10, 15, 100, 105, 110, 200, 205, 210]
    input_spikes = np.zeros(T, dtype=np.float32)
    input_spikes[input_spikes_list] = 1

    t_values = np.linspace(0, 2 * np.pi, T)

    for t in range(1, T):

        V_lif[t] = alpha * V_lif[t - 1]

        resonator_activation = 0.1 * np.sin(freq * t_values[t])
        V_res[t] = alpha * V_res[t - 1] + resonator_activation

        if t in input_spikes_list:
            V_lif[t] = V_lif[t - 1] + 0.3
            V_res[t] = V_res[t - 1] + 0.3


        V_lif_plot[t] = V_lif[t]
        V_res_plot[t] = V_res[t]

        if V_lif[t] > 1:
            V_lif[t] = 0
            spk_lif[t] = 1

        if V_res[t] > 1:
            V_res[t] = 0
            spk_res[t] = 1


    #
    # Plot
    #

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3, 1]}
    )

   
    axes[0].eventplot(
        np.where(input_spikes)[0],
        colors="black",
        lineoffsets=0,
        linelengths=0.8
    )
    axes[0].set_ylabel("Input")
    axes[0].set_yticks([])


    axes[1].plot(V_lif_plot, label="V_lif", linewidth=2)
    axes[1].plot(V_res_plot, label="V_res", linewidth=2)
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.6, label="Threshold")
    axes[1].set_ylabel("Membrane Potential")
    axes[1].legend()

  
    axes[2].eventplot(
        [
            np.where(spk_lif)[0],
            np.where(spk_res)[0],
        ],
        colors=["tab:blue", "tab:orange"],
        lineoffsets=[1, 0],
        linelengths=0.8
    )
    
    axes[2].set_yticks([1, 0])
    axes[2].set_yticklabels(["LIF", "Res"])
    axes[2].set_ylabel("Output")
    axes[2].set_xlabel("Time step")


    plt.tight_layout()
    plt.savefig("./plots/CompareIntRes.png", dpi=200)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")