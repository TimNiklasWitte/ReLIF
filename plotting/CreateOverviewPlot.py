from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

def main():

    T = 300
    alpha = 0.9
    freq = 5

    V_res = np.zeros(T, dtype=np.float32)

    V_res_plot = np.zeros(T, dtype=np.float32)
    resonator_activation_plot = np.zeros(T, dtype=np.float32)

    spk_res = np.zeros(T, dtype=np.float32)

    input_spikes_list = [10, 15, 100, 105, 110, 200, 205, 210]
    input_spikes = np.zeros(T, dtype=np.float32)
    input_spikes[input_spikes_list] = 1

    t_values = np.linspace(0, 2 * np.pi, T)

    for t in range(1, T):


        resonator_activation = 0.5 * np.sin(freq * t_values[t])

        resonator_activation_plot[t] = resonator_activation

        V_res[t] = alpha * V_res[t - 1]

        if t in input_spikes_list:
            V_res[t] = V_res[t - 1] + 0.4

        V_res_plot[t] = V_res[t]

        if V_res[t] + resonator_activation > 1:
            V_res[t] = 0
            spk_res[t] = 1


    #
    # Plot
    #

    fig, axes = plt.subplots(
        5, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 3, 2, 1]}
    )

    # Input spikes
    axes[0].eventplot(
        np.where(input_spikes)[0],
        colors="black",
        lineoffsets=0,
        linelengths=0.8,
    )
    axes[0].set_ylabel("Input")
    axes[0].set_yticks([])

    # Resonator drive
    axes[1].plot(resonator_activation_plot, color="tab:green", linewidth=2)
    axes[1].set_ylabel("resonator_activation")
    axes[1].grid(True)



    # Membrane potential
    axes[2].plot(V_res_plot, color="tab:blue", linewidth=2, label="V")
    #axes[2].axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Threshold")
    axes[2].set_ylabel(r"$V_t$")
    axes[2].legend()
    axes[2].grid(True)

    V_total = V_res_plot + resonator_activation_plot
    # Membrane potential
    axes[3].plot(V_total, color="tab:blue", linewidth=2, label="V")
    axes[3].axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Threshold")
    axes[3].set_ylabel(r"$V_t$")
    axes[3].legend()
    axes[3].grid(True)

    # Output spikes
    axes[4].eventplot(
        np.where(spk_res)[0],
        colors="tab:orange",
        lineoffsets=0,
        linelengths=0.8,
    )
    axes[4].set_ylabel("Spikes")
    axes[4].set_yticks([])
    axes[4].set_xlabel("Time step")

    plt.tight_layout()
    plt.savefig("./plots/Overview.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")