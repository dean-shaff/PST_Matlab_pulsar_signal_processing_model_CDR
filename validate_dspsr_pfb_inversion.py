import os
import sys

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pfb.util import load_dada_file
from pfb.rational import Rational
from pfb.formats import DADAFile

base_dir = os.path.dirname(os.path.abspath(__file__))

fft_size = 229466

# vanilla_dspsr_dump = DADAFile("pre_Fold.vanilla.pulsar.10.dump")
# pfb_inversion_os_dump = DADAFile("pre_Fold.py.inverse.pulsar.10.os.dump")

vanilla_dspsr_dump = DADAFile("pre_Detection.vanilla.pulsar.10.dump")
pfb_inversion_os_dump = DADAFile("pre_Detection.py.inverse.pulsar.10.os.dump")

vanilla_dspsr_dump.load_data()
print(vanilla_dspsr_dump.data.shape)
pfb_inversion_os_dump.load_data()

min_size = min(vanilla_dspsr_dump.ndat, pfb_inversion_os_dump.ndat)
nseries = int(min_size // fft_size)
nseries = int(nseries / 4)
os_factor = Rational(*pfb_inversion_os_dump["OS_FACTOR"].split("/"))
# pfb_inversion_os_dump._data /= float(os_factor)**2
xlim = [0, nseries*fft_size]
idx = slice(*xlim)

def get_time_shift(a, b):
    a = a.copy()
    b = b.copy()
    a /= np.amax(a)
    b /= np.amax(b)
    xcorr = scipy.signal.fftconvolve(a, np.conj(b)[::-1], mode="full")
    mid_idx = int(xcorr.shape[0] // 2)
    max_arg = np.argmax(xcorr)
    offset = max_arg - mid_idx

    return xcorr, offset


def compare_plot_time_series():

    fig, axes = plt.subplots(4, 4, figsize=(24,24))

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].grid(True)
        # axes[i].set_xlim(xlim)
    ave_errors = []
    max_errors = []
    for ipol in range(2):
        for iz in range(2):
            pol_z_idx = ipol*2 + iz
            # z_func = np.abs if iz == 0 else np.angle
            # z_text = "Power spectrum" if iz == 0 else "Phase"
            z_func = np.real if iz == 0 else np.imag
            z_text = "Real" if iz == 0 else "Imaginary"
            van_dat = z_func(vanilla_dspsr_dump.data[idx,0,ipol])
            van_dat /= np.amax(van_dat)
            inv_dat = z_func(pfb_inversion_os_dump.data[idx,0,ipol])
            inv_dat /= np.amax(inv_dat)

            van = 0
            argmax = np.argmax(van_dat)
            axes[van, pol_z_idx].plot(van_dat, color="green")
            # axes[van].axvline(argmax, color="green")
            # axes[van].plot(vanilla_dspsr_dump.data[idx,0,0,1])
            axes[van, pol_z_idx].set_title(f"{z_text} Pol {ipol} Vanilla dspsr")

            inv = 1
            argmax = np.argmax(inv_dat)
            axes[inv, pol_z_idx].plot(inv_dat, color="green")
            axes[inv, pol_z_idx].set_title(f"{z_text} Pol {ipol} PFB Inversion")

            # axes[inv].axvline(argmax, color="green")
            # axes[inv].plot(pfb_inversion_os_dump.data[idx,0,0,1])
            xcorr, offset = get_time_shift(van_dat, inv_dat)

            axes[inv + 1, pol_z_idx].plot(xcorr)
            axes[inv + 1, pol_z_idx].set_title(f"Cross Correlation: Offset={offset}")

            diff = np.roll(van_dat, abs(offset)) - inv_dat
            axes[inv + 2, pol_z_idx].set_yscale('log')
            axes[inv + 2, pol_z_idx].plot(np.abs(diff))
            axes[inv + 2, pol_z_idx].set_title(f"Offest corrected difference")
            ave_errors.append(np.mean(np.abs(diff)))
            max_errors.append(np.amax(np.abs(diff)))
            print(f"Average error: {ave_errors[-1]:.7f}, max error: {max_errors[-1]:.7f}")

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    fig.savefig("validate_pfb_inversion.time_series.png")

def compare_plot_frequency():
    fig, axes = plt.subplots(3, 4, figsize=(24,24))

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].grid(True)
        # axes[i].set_xlim(xlim)
    ave_errors = []
    max_errors = []
    for ipol in range(2):
        for iz in range(2):
            pol_z_idx = ipol*2 + iz
            # z_func = np.abs if iz == 0 else np.angle
            # z_text = "Power spectrum" if iz == 0 else "Phase"
            z_func = np.abs if iz == 0 else np.angle
            z_text = "Power Spectrum" if iz == 0 else "Phase"

            van_dat = vanilla_dspsr_dump.data[idx,0,ipol]
            inv_dat = pfb_inversion_os_dump.data[idx,0,ipol]

            xcorr, offset = get_time_shift(van_dat, inv_dat)
            print(f"offset={offset}")
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
            axes[van, pol_z_idx].set_title(f"{z_text} Pol {ipol} Vanilla dspsr")

            inv = 1
            argmax = np.argmax(inv_dat_spec)
            axes[inv, pol_z_idx].plot(inv_dat_spec, color="green")
            axes[inv, pol_z_idx].set_title(f"{z_text} Pol {ipol} PFB Inversion")

            diff = van_dat_spec - inv_dat_spec

            axes[inv + 1, pol_z_idx].set_yscale('log')
            axes[inv + 1, pol_z_idx].plot(np.abs(diff))
            axes[inv + 1, pol_z_idx].set_title(f"Offest corrected difference")
            ave_errors.append(np.mean(np.abs(diff)))
            max_errors.append(np.amax(np.abs(diff)))
            print(f"Average error: {ave_errors[-1]:.7f}, max error: {max_errors[-1]:.7f}")

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    fig.savefig("validate_pfb_inversion.frequency.png")


if __name__ == "__main__":
    compare_plot_time_series()
    # compare_plot_frequency()
    plt.show()
