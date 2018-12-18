import argparse

import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt


def create_parser():
    parser = argparse.ArgumentParser(
        description=("Plot the Finite Impulse Response "
                     "stored in a MATLAB .mat file"))
    parser.add_argument("-i", "--in-file", dest="file_path_in")
    return parser


def main():

    parsed = create_parser().parse_args()

    fir = scipy.io.loadmat(parsed.file_path_in)
    try:
        nchan = fir['Nc'][0][0]
    except KeyError:
        nchan = fir['Nchan'][0][0]
    cut_off = fir['Fp'][0][0]
    stop_band = fir['Fs'][0][0]
    As, Ap = np.int64(fir['As'][0][0]), fir['Ap'][0][0]
    print(f"As: {As}")
    print(f"Ap: {Ap}")

    filt = fir["h"][0]
    w, h = scipy.signal.freqz(filt, 1, int(filt.shape[0]*nchan))
    w /= np.pi

    fig, axes = plt.subplots(3, 1)
    for ax in axes:
        ax.grid(True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.80])

    axes[0].plot(w, np.abs(h))
    axes[0].set_title("Transfer function of filter")
    # axes[0].set_xlim([0, 3.5*cut_off])
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([-0.15, 1.15])

    n_points = 100
    pass_band_x = np.linspace(0, cut_off, n_points)
    axes[1].plot(w, 20*np.log10(np.abs(h)))
    axes[1].plot(pass_band_x, -0.5*Ap*np.ones(n_points), color="k")
    axes[1].plot(pass_band_x, 0.5*Ap*np.ones(n_points), color="k")
    # axes[1].set_xlim([0, 1.5*cut_off])
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([-4.5*Ap, 4.5*Ap])
    axes[1].set_title("Passband")

    stop_band_x = np.linspace(stop_band, 1, n_points)
    stop_band_y = -As*np.ones(n_points)
    axes[2].plot(w, 20*np.log10(np.abs(h)))
    axes[2].plot(stop_band_x, stop_band_y, color="r")
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([-(As+10), 3])
    axes[2].set_title("Stopband")

    fig.suptitle(
        (f"Filter Coefficients\n"
         f"Number of Channels {nchan}\n"
         f"Cut off frequency {cut_off}\n"
         f"Stop-band frequency {stop_band}"))

    plt.show()


if __name__ == "__main__":
    main()
