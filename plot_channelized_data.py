import csv

import numpy as np
import matplotlib.pyplot as plt

files = ["pol1.csv", "pol2.csv"]


def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(np.asarray([float(i) for i in line]))
    return np.asarray(data)


def main():

    fig, axs = plt.subplots(2, 1)
    pol1_data, pol2_data = [read_data(n) for n in files]

    pol1_data = pol1_data[:, 0] + 1j* pol1_data[:, 1]
    pol2_data = pol2_data[:, 0] + 1j* pol2_data[:, 1]

    axs[0].plot(np.abs(pol1_data)**2)
    axs[1].plot(np.abs(pol2_data)**2)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
