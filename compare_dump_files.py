# compare_dump_files.py
import argparse

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pfb_channelizer import PFBChannelizer


def compare_dump_files(file_path0, file_path1, **kwargs):
    comp_dat = []
    dat_sizes = np.zeros(2)
    fnames = [file_path0, file_path1]
    for i, fname in enumerate(fnames):
        with open(fname, "rb") as input_file:
            buffer = input_file.read()
            header = np.frombuffer(
                buffer, dtype='c', count=PFBChannelizer.header_size)
            data = np.frombuffer(
                buffer, dtype=PFBChannelizer.input_dtype,
                offset=PFBChannelizer.header_size)
        dat_sizes[i] = data.shape[0]
        comp_dat.append(data)
    min_size = int(np.amin(dat_sizes))
    comp_dat = [d[:min_size] for d in comp_dat]
    xlim = [1040, 1060]
    fig, axes = plt.subplots(len(fnames)+2)
    fig.tight_layout()
    # fig.tight_layout(rect=[0, 0.03, 0, 0.95])
    for ax in axes:
        ax.grid(True)
        # ax.set_xlim(xlim)

    axes[0].plot(comp_dat[0], alpha=0.7)
    axes[0].set_title("Signal 1")
    axes[1].plot(comp_dat[1], alpha=0.7)
    axes[1].set_title("Signal 2")
    diff_squared = (comp_dat[0] - comp_dat[1])**2
    axes[2].plot(diff_squared, alpha=0.7)
    axes[2].set_title("Squared difference")
    x_corr_left = scipy.signal.fftconvolve(
        comp_dat[0], comp_dat[1][::-1], mode="same")
    axes[3].plot(x_corr_left)
    axes[3].set_title("Cross correlation")

    if not np.allclose(*comp_dat, **kwargs):
        print("dump files are not the same")
        print((f"{np.sum(diff_squared == 0.0)}/"
               f"{diff_squared.shape[0]} are the same."))
    else:
        print("dump files are the same")

    plt.show()


def create_parser():

    parser = argparse.ArgumentParser(
        description="compare the contents of two dump files")

    parser.add_argument("-i", "--input-files",
                        dest="input_file_paths",
                        nargs="+", type=str,
                        required=True)

    parser.add_argument("-v", "--verbose",
                        dest="verbose", action="store_true")

    return parser


if __name__ == '__main__':
    parsed = create_parser().parse_args()
    compare_dump_files(*parsed.input_file_paths)
