"""
validate_matlab_scipy_freqz.py

In order to obtain the frequency response of an FIRFilter in dspsr, I need
to be able to implement an equivalent function to matlab's `freqz`
or scipy's `scipy.signal.freqz`. Matlab's `freqz` function is used in
PFBinversion in order to apply derippling.

This script simply determines if matlab and scipy do the same thing when
calling `freqz`
"""
import os

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt

cur_dir = os.path.dirname(os.path.abspath(__file__))

fir_filter_file_path = os.path.join(cur_dir, "config", "Prototype_FIR.mat")
matlab_freqz_file_path = os.path.join(cur_dir, "config", "TF_points.mat")

plot = False

def main():

    fir_obj = scipy.io.loadmat(fir_filter_file_path)
    fir = fir_obj["h"].transpose()
    matlab_freqz_obj = scipy.io.loadmat(matlab_freqz_file_path)
    matlab_freqz = matlab_freqz_obj["H0"]

    w, h = scipy.signal.freqz(fir, 1, worN=len(matlab_freqz), whole=False)
    matlab_freqz = matlab_freqz.reshape(h.shape)

    if plot:
        fig, axes = plt.subplots(2, 1)

        axes[0].plot(matlab_freqz)
        axes[0].set_title("Matlab freqz")

        axes[1].plot(h)
        axes[1].set_title("scipy freqz")

        plt.show()
    allclose = np.allclose(h, matlab_freqz)
    print("Are matlab and scipy computing same thing? {}".format(
        "yes" if allclose else "no"))


if __name__ == "__main__":
    main()
