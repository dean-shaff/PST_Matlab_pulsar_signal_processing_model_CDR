import csv

import numpy as np
import matplotlib.pyplot as plt

files = ["pol1_channel1_series1.csv", "pol2_channel1_series1.csv"]

def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(np.asarray([float(i) for i in line]))
    return np.asarray(data)


def main():
    chan_num = 2
    n_bins = 16384
    fig, axs = plt.subplots(2, 1)
    pol1, pol2 = [read_data(n) for n in files]

    pol1 = pol1[:, 0] + 1j* pol1[:, 1]
    pol2 = pol2[:, 0] + 1j* pol2[:, 1]
    # axs[0].plot(np.real(pol1[chan_num*n_bins:(chan_num + 1)*n_bins]))
    # axs[1].plot(np.imag(pol1[chan_num*n_bins:(chan_num + 1)*n_bins]))
    axs[0].plot(np.real(pol1))
    axs[1].plot(np.imag(pol1))

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
