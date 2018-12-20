# compare_dump_files.py
import argparse
import os

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pfb_channelizer import PFBChannelizer


def plot_dump_files(*file_paths, no_header=False, complex=False):
    comp_dat = []
    dat_sizes = np.zeros(len(file_paths))
    header_offset = PFBChannelizer.header_size
    if no_header:
        header_offset = 0
    for i, fname in enumerate(file_paths):
        with open(fname, "rb") as input_file:
            buffer = input_file.read()
            # header = np.frombuffer(
            #     buffer, dtype='c', count=PFBChannelizer.header_size)
            data = np.frombuffer(
                buffer, dtype=PFBChannelizer.input_dtype,
                offset=header_offset)
            if complex:
                data = data.reshape((-1, 2))
                data = data[:, 0] + 1j*data[:, 1]
                # data = data.reshape((2, -1))
                # data = data[0, :] + 1j*data[1, :]

        dat_sizes[i] = data.shape[0]
        comp_dat.append(data)
    # min_size = int(np.amin(dat_sizes))
    fig, axes = plt.subplots(len(file_paths), len(file_paths))
    fig.tight_layout()

    if not hasattr(axes, "__iter__"):
        axes.grid(True)
        axes.plot(comp_dat[0])
        axes.set_title(os.path.basename(file_paths[0]))
    else:
        for row in range(axes.shape[0]):
            for col in range(axes.shape[1]):
                ax = axes[row, col]
                ax.grid(True)
                if row == col:
                    ax.grid(True)
                    ax.plot(comp_dat[i])
                    ax.set_title(os.path.basename(file_paths[row]))
                elif row > col:
                    x_corr = scipy.signal.fftconvolve(
                        comp_dat[row],
                        comp_dat[col][::-1],
                        mode="same"
                    )
                    ax.plot(x_corr)
                    ax.set_title(
                        (f"cross correlation:\n"
                         f"{os.path.basename(file_paths[row])}\n"
                         f"{os.path.basename(file_paths[col])}"))
                else:
                    ax.set_axis_off()
    plt.show()


def create_parser():

    parser = argparse.ArgumentParser(
        description="compare the contents of two dump files")

    parser.add_argument("-i", "--input-files",
                        dest="input_file_paths",
                        nargs="+", type=str,
                        required=True)

    parser.add_argument('-nh', "--no-header",
                        dest="no_header", action="store_true")

    parser.add_argument('-c', "--complex",
                        dest="complex", action="store_true")

    parser.add_argument("-v", "--verbose",
                        dest="verbose", action="store_true")

    return parser


if __name__ == '__main__':
    parsed = create_parser().parse_args()
    plot_dump_files(
        *parsed.input_file_paths,
        no_header=parsed.no_header,
        complex=parsed.complex
    )
