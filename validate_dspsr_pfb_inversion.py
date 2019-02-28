import os
import argparse
import typing
from contextlib import contextmanager

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pfb.formats import DADAFile

base_dir = os.path.dirname(os.path.abspath(__file__))


def get_time_shift(a: np.ndarray,
                   b: np.ndarray) -> typing.Tuple[np.ndarray, int]:
    a = a.copy()
    b = b.copy()
    a /= np.amax(a)
    b /= np.amax(b)
    xcorr = scipy.signal.fftconvolve(a, np.conj(b)[::-1], mode="full")
    mid_idx = int(xcorr.shape[0] // 2)
    max_arg = np.argmax(xcorr)
    offset = max_arg - mid_idx

    return xcorr, offset


def get_nseries(*dada_objs: typing.Tuple[DADAFile],
                fft_size: int = 229376) -> int:
    min_size = min(*[obj.ndat for obj in dada_objs])
    nseries = int(min_size // fft_size)
    return nseries


@contextmanager
def compare(*dada_objs: typing.Tuple[DADAFile],
            name: str = "frequency",
            fft_size: int = 229376,
            n_samples: float = 1.0) -> None:

    fig, axes = plt.subplots(4, 4, figsize=(24, 24))

    nseries = get_nseries(*dada_objs, fft_size=fft_size)
    nseries = int(nseries * n_samples)

    xlim = [0, nseries*fft_size]

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].grid(True)
    plot_obj = {
        "dada_objs": dada_objs,
        "fft_size": fft_size,
        "nseries": nseries,
        "xlim": xlim,
        "idx": slice(*xlim),
        "axes": axes,
        "ave_errors": [],
        "max_errors": []
    }

    try:
        yield plot_obj

    finally:
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        no_ext = [os.path.splitext(os.path.basename(obj.file_path))[0]
                  for obj in dada_objs]
        output_file_name = "validate_pfb_inversion.{}.{}.{}".format(
            name, *no_ext)
        fig.savefig(output_file_name + ".png")
        with open(output_file_name + ".txt", "w") as f:
            f.write("Ave, Max\n")
            for i in range(len(plot_obj["ave_errors"])):
                output_str_list = [f"{plot_obj['ave_errors'][i]:.9f}",
                              f"{plot_obj['max_errors'][i]:.9f}"]
                f.write(", ".join(output_str_list) + "\n")
                print("Average error: {}, max error: {}".format(
                    *output_str_list
                ))


def compare_plot_time_series(plot_obj: dict) -> None:

    vanilla_dspsr_dump, pfb_inversion_dump = plot_obj["dada_objs"]
    axes = plot_obj["axes"]

    for ipol in range(2):
        for iz in range(2):
            pol_z_idx = ipol*2 + iz
            # z_func = np.abs if iz == 0 else np.angle
            # z_text = "Power spectrum" if iz == 0 else "Phase"
            z_func = np.real if iz == 0 else np.imag
            z_text = "Real" if iz == 0 else "Imaginary"
            van_dat = z_func(vanilla_dspsr_dump.data[plot_obj["idx"], 0, ipol])
            van_dat /= np.amax(van_dat)
            inv_dat = z_func(pfb_inversion_dump.data[plot_obj["idx"], 0, ipol])
            inv_dat /= np.amax(inv_dat)

            van = 0
            argmax = np.argmax(van_dat)
            axes[van, pol_z_idx].plot(van_dat, color="green")
            # axes[van].axvline(argmax, color="green")
            # axes[van].plot(vanilla_dspsr_dump.data[idx,0,0,1])
            axes[van, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} Vanilla dspsr")

            inv = 1
            argmax = np.argmax(inv_dat)
            axes[inv, pol_z_idx].plot(inv_dat, color="green")
            axes[inv, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} PFB Inversion")

            # axes[inv].axvline(argmax, color="green")
            # axes[inv].plot(pfb_inversion_dump.data[idx,0,0,1])
            xcorr, offset = get_time_shift(van_dat, inv_dat)

            axes[inv + 1, pol_z_idx].plot(xcorr)
            axes[inv + 1, pol_z_idx].set_title(
                f"Cross Correlation: Offset={offset}")

            diff = np.roll(van_dat, abs(offset)) - inv_dat
            axes[inv + 2, pol_z_idx].set_yscale('log')
            axes[inv + 2, pol_z_idx].plot(np.abs(diff))
            axes[inv + 2, pol_z_idx].set_title(f"Offest corrected difference")
            plot_obj["ave_errors"].append(np.mean(np.abs(diff)))
            plot_obj["max_errors"].append(np.amax(np.abs(diff)))
            # print((f"Average error: {ave_errors[-1]:.9f}, "
            #        f"max error: {max_errors[-1]:.9f}"))


def compare_plot_frequency(plot_obj: dict) -> None:

    vanilla_dspsr_dump, pfb_inversion_dump = plot_obj["dada_objs"]
    axes = plot_obj["axes"]
    fft_size = plot_obj["fft_size"]

    for ipol in range(2):
        for iz in range(2):
            pol_z_idx = ipol*2 + iz
            # z_func = np.abs if iz == 0 else np.angle
            # z_text = "Power spectrum" if iz == 0 else "Phase"
            z_func = np.abs if iz == 0 else np.angle
            z_text = "Power Spectrum" if iz == 0 else "Phase"

            van_dat = vanilla_dspsr_dump.data[plot_obj["idx"], 0, ipol]
            inv_dat = pfb_inversion_dump.data[plot_obj["idx"], 0, ipol]

            xcorr, offset = get_time_shift(van_dat, inv_dat)
            # print(f"offset={offset}")
            van_dat = np.roll(van_dat, abs(offset))

            van_dat_spec = z_func(np.fft.fft(van_dat[fft_size:2*fft_size]))
            van_dat_spec /= np.amax(van_dat_spec)
            inv_dat_spec = z_func(np.fft.fft(inv_dat[fft_size:2*fft_size]))
            inv_dat_spec /= np.amax(inv_dat_spec)

            van = 0
            argmax = np.argmax(van_dat_spec)
            axes[van, pol_z_idx].plot(van_dat_spec, color="green")
            # axes[van].axvline(argmax, color="green")
            # axes[van].plot(vanilla_dspsr_dump.data[idx,0,0,1])
            axes[van, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} Vanilla dspsr")

            inv = 1
            argmax = np.argmax(inv_dat_spec)
            axes[inv, pol_z_idx].plot(inv_dat_spec, color="green")
            axes[inv, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} PFB Inversion")

            diff = van_dat_spec - inv_dat_spec

            axes[inv + 1, pol_z_idx].set_yscale('log')
            axes[inv + 1, pol_z_idx].plot(np.abs(diff))
            axes[inv + 1, pol_z_idx].set_title(f"Offest corrected difference")
            plot_obj["ave_errors"].append(np.mean(np.abs(diff)))
            plot_obj["max_errors"].append(np.amax(np.abs(diff)))


def create_parser():

    parser = argparse.ArgumentParser(
        description="Comapare dspsr dump files")

    parser.add_argument("-i", "--input-files",
                        dest="input_file_paths",
                        nargs="+", type=str,
                        required=True)

    parser.add_argument('-x', "--fft_size",
                        dest="fft_size", type=int, default=229376)

    # parser.add_argument('-nh', "--no-header",
    #                     dest="no_header", action="store_true")
    #
    # parser.add_argument('-c', "--complex",
    #                     dest="complex", action="store_true")
    #
    parser.add_argument('-n', "--n-samples",
                        dest="n_samples", type=float)
    #
    # parser.add_argument(
    #     '-op', "--operation",
    #     dest="op", type=str,
    #     help=(f"Apply 'op' to each pair of input files. "
    #           f"Available operations: {list(op_lookup.keys())}"))

    parser.add_argument("-v", "--verbose",
                        dest="verbose", action="store_true")

    return parser


def main():
    parsed = create_parser().parse_args()
    vanilla_dspsr_dump = DADAFile(parsed.input_file_paths[0])
    pfb_inversion_dump = DADAFile(parsed.input_file_paths[1])

    vanilla_dspsr_dump.load_data()
    pfb_inversion_dump.load_data()

    # with compare(
    #     vanilla_dspsr_dump,
    #     pfb_inversion_dump,
    #     name="frequency",
    #     fft_size=parsed.fft_size,
    #     n_samples=parsed.n_samples
    # ) as plot_obj:
    #     compare_plot_frequency(plot_obj)

    with compare(
        vanilla_dspsr_dump,
        pfb_inversion_dump,
        name="time_series",
        fft_size=parsed.fft_size,
        n_samples=parsed.n_samples
    ) as plot_obj:
        compare_plot_time_series(plot_obj)


if __name__ == "__main__":
    main()
