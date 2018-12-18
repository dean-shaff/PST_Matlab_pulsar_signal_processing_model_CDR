import os
import time
import logging
import multiprocessing

import numba
import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt

from rational import Rational

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "config")
data_dir = os.path.join(current_dir, "data")

module_logger = logging.getLogger(__name__)

def coroutine(func):
    def start(*args, **kwargs):
        coro = func(*args, **kwargs)
        next(coro)
        return coro
    return start

class Consumer:

    def __init__(self):
        self.val = None

    def send(self, val):
        self.val = val


def process_header(header_arr):
    header_str = "".join([c.decode("UTF-8") for c in header_arr.tolist()])
    lines = header_str.split("\n")
    header = {}
    for line in lines:
        if line.startswith("#") or not line:
            continue
        else:
            key, val = line.split()[:2]
            try:
                val = float(val)
            except ValueError as err:
                pass
            header[key] = val
    return header


def load_dump_data(file_path, header_size=4096, float_dtype=np.float32):
    with open(file_path, "rb") as file:
        buffer = file.read()
        header = np.frombuffer(
            buffer, dtype='c', count=header_size)
        data = np.frombuffer(
            buffer, dtype=float_dtype, offset=header_size)
    return [header, data]

# @numba.njit
@numba.njit(numba.float32[:,:](numba.float32[:], numba.float32[:], numba.float32[:,:], numba.int64))
def filter(signal, filter_coef, filtered, downsample_by):
    """
    filter signal with filter_coef.
    """
    rem = filter_coef.shape[0] % downsample_by
    if rem != 0:
        filter_coef_padded = np.zeros((filter_coef.shape[0] + downsample_by - rem), dtype=np.float32)
        filter_coef_padded[filter_coef.shape[0]:] = filter_coef
        filter_coef = filter_coef_padded

    down_sample_filter_elem = int(filter_coef.shape[0] / downsample_by)
    filter_idx = np.arange(filter_coef.shape[0]).reshape(
        (down_sample_filter_elem, downsample_by))
    # filter_coeff_2d = filter_coef[filter_idx]
    filter_coeff_2d = np.zeros(filter_idx.shape, dtype=filter_coef.dtype)
    for i in range(downsample_by):
        filter_coeff_2d[:,i] = filter_coef[filter_idx[:, i]]

    window_size = filter_coef.shape[0]
    signal_padded = np.zeros((window_size + signal.shape[0]), dtype=np.float32)
    signal_padded[window_size:] = signal

    # if filtered is None:
    #     rem = signal_padded.shape[0] % window_size
    #     if rem != 0:
    #         signal_padded = np.append(signal_padded, np.zeros(window_size - rem))
    #     down_sample_signal_elem = int(signal_padded.shape[0]/downsample_by) - down_sample_filter_elem
    #     filtered = np.zeros((down_sample_signal_elem, downsample_by))
    # else:

    down_sample_signal_elem = filtered.shape[0]
    signal_chunk_2d = np.zeros(filter_idx.shape, dtype=signal_padded.dtype)

    for i in range(down_sample_signal_elem):
        idx = i*downsample_by
        signal_chunk = signal_padded[idx:idx + window_size][::-1]
        # filtered[i,:] = np.sum(signal_chunk[filter_idx] * filter_coeff_2d, axis=0)
        for j in range(downsample_by):
            signal_chunk_2d[:, j] = signal_chunk[filter_idx[:, j]]
        filtered[i, :]  = np.sum(signal_chunk_2d * filter_coeff_2d, axis=0)

    return filtered


class PFBChannelizer:

    header_size = 4096
    default_input_samples = 2**14
    pfb_fir_config_file_path = os.path.join(config_dir, "OS_Prototype_FIR_8.mat")
    complex_dtype = np.complex64
    float_dtype = np.float32


    def __init__(self, input_file_path, oversampling_factor, output_channels=8, output_file_path=None):
        self.logger = module_logger.getChild("PFBChannelizer")
        self.input_file_path = input_file_path
        self.oversampling_factor = oversampling_factor
        self.oversampled = False
        if float(self.oversampling_factor) != 1.0:
            self.oversampled = True

        if output_file_path is None:
            os_text = "cs"
            if self.oversampled:
                os_text = "os"
            output_file_path = self.input_file_path.replace("simulated_pulsar", "py_channelized")
            split = output_file_path.split(".")
            split.insert(-1, os_text)
            output_file_path = ".".join(split)

        self.output_file_path = output_file_path
        self.logger.debug(f"__init__: input_file_path: {input_file_path}")
        self.logger.debug(f"__init__: output_file_path: {output_file_path}")

        self.input_data = None
        self.input_header = None
        self._input_npol = None
        self._input_nchan = 1
        self._input_ndim = 1

        self.output_data = None
        self.output_header = None
        self._output_npol = 2
        self._output_nchan = output_channels
        self._output_ndim = 2

        self._n_series = None
        self._ndim_ratio = Rational(1, 1)
        self._input_samples = self.default_input_samples # n_in changes due to oversampling
        self._output_samples = 0

        self._pfb_config = None
        self._pfb_filter_coef = None

    def _load_pfb_config(self, diagnostic_plots=False):

        t0 = time.time()
        self._pfb_config = scipy.io.loadmat(self.pfb_fir_config_file_path)
        self._pfb_filter_coef = self._pfb_config["h"].reshape(-1)
        self._pfb_filter_coef = self._pfb_filter_coef.astype(self.float_dtype)

        if diagnostic_plots:
            plt.ion()
            fig, ax = plt.subplots()
            ax.grid(True)
            ax.plot(self._pfb_filter_coef)
            input(">> ")

        rem = self._pfb_filter_coef.shape[0] % self._output_nchan
        if rem != 0:
            self._pfb_filter_coef = np.append(
                self._pfb_filter_coef, np.zeros(self._output_nchan - rem, dtype=self.float_dtype))

        input_mask_dtype = self.float_dtype
        if self._input_ndim == 2:
            input_mask_dtype = self.complex_dtype

        self._pfb_input_mask = np.zeros(
            (self._output_npol, self._pfb_filter_coef.shape[0]),
            dtype=input_mask_dtype
        )

        self._pfb_output_mask = np.zeros(
            (self._output_npol, self._output_nchan), dtype=self.float_dtype)
        self.logger.debug(
            (f"_load_pfb_config: self._pfb_filter_coef.dtype: "
             f" {self._pfb_filter_coef.dtype}"))
        self.logger.debug(
            (f"_load_pfb_config: self._pfb_filter_coef.shape:"
             f" {self._pfb_filter_coef.shape}"))
        self.logger.debug(
            (f"_load_pfb_config: Took {time.time()-t0:.4f} "
             f"seconds to load pfb configuration data"))

    def _load_input_data(self):

        t0 = time.time()
        header, data = load_dump_data(
            self.input_file_path,
            header_size=self.header_size,
            float_dtype=self.float_dtype
        )
        self.logger.debug(
            f"_load_input_data: Took {time.time()-t0:.4f} seconds to load input data")

        self.input_header = process_header(header)
        self.input_data = data

        self._input_npol = int(self.input_header["NPOL"])
        self._input_ndim = int(self.input_header["NDIM"])
        self._input_nchan = int(self.input_header["NCHAN"])
        self._ndim_ratio = Rational(self._output_ndim, self._input_ndim)

    def _init_output_data(self):

        if self.input_header is None:
            raise RuntimeError("Need to load input header before initializing output data")

        ndat_input = self.input_data.shape[0]
        norm_chan = self.oversampling_factor.normalize(self._output_nchan)
        int_ndim_ratio = int(self._ndim_ratio)
        self._output_samples = int(ndat_input / (norm_chan * int_ndim_ratio))
        self._input_samples = self._output_samples * norm_chan * int_ndim_ratio

        self.logger.debug(f"_init_output_data: ndat_input: {ndat_input}")
        self.logger.debug(f"_init_output_data: self._output_samples: {self._output_samples}")
        self.logger.debug(f"_init_output_data: self._input_samples: {self._input_samples}")

        self.output_data = np.zeros((
            int(self._output_samples / (self._input_npol * self._input_ndim)),
            self._output_nchan,
            self._output_ndim * self._output_npol
        ), dtype=self.float_dtype)


        # self._input_samples = int(self.default_input_samples * norm_chan)
        # self._output_samples = int(self._input_samples / norm_chan /int(self._ndim_ratio))
        # self._n_series = int(ndat_input//(self._input_npol*self._input_samples*self._input_ndim))
        #
        # self.output_data = np.zeros(
        #      (int(self._n_series*self._output_samples),
        #       self._output_nchan,
        #       self._output_npol*self._output_ndim),
        #      dtype=self.float_dtype
        # )

    def _init_output_header(self):

        if self.input_header is None:
            raise RuntimeError("Need to load input header before initializing output header")

        os_factor = self.oversampling_factor

        self.output_header = self.input_header.copy()
        self.output_header['NDIM'] = 2
        self.output_header['NCHAN'] = self._output_nchan
        self.output_header['NPOL'] = self._output_npol
        # have to adjust TSAMP
        input_tsamp = self.input_header["TSAMP"]
        self.logger.debug(f"_init_output_header: input_tsamp: {input_tsamp}")
        output_tsamp = (
            int(self._ndim_ratio) *
            input_tsamp *
            self._output_nchan *
            os_factor.de/os_factor.nu
        )
        self.logger.debug(f"_init_output_header: output_tsamp: {output_tsamp}")
        self.output_header['TSAMP'] = output_tsamp
        self.output_header['OS_FACTOR'] = f"{os_factor.nu}/{os_factor.de}"

    def _dump_data(self, header, data, **kwargs):

        self.logger.debug(f"_dump_data: header: {header}")
        t0 = time.time()

        header_str = "\n".join(
            [f"{key} {header[key]}" for key in header]) + "\n"
        header_bytes = str.encode(header_str)
        remaining_bytes = self.header_size - len(header_bytes)
        self.logger.debug(
            f"_dump_data: len(header_bytes): {len(header_bytes)}")
        self.logger.debug(
            f"_dump_data: remaining_bytes: {remaining_bytes}")
        header_bytes += str.encode(
            "".join(["\0" for i in range(remaining_bytes)]))

        assert len(header_bytes) == self.header_size, \
            f"Number of bytes in header must be equal to {self.header_size}"

        with open(self.output_file_path, "wb") as output_file:
            output_file.write(header_bytes)
            output_file.write(data.flatten(**kwargs).tobytes())

        self.logger.debug(
            f"_dump_data: Took {time.time() - t0:.4f} seconds to dump data")


    @coroutine
    def _pfb(self, ipol, sink=None):
        """
        Coroutine for computing polyphase filterbank.
        """
        # index to keep track of total number of iterations
        n_iter = 0
        # control index for oversampling case
        n_os_ctrl = 0
        # precompute some values that will get used heavily in the main loop
        nchan = self._output_nchan
        nchan_norm = self.oversampling_factor.normalize(nchan)
        nchan_overlap = nchan - nchan_norm
        nchan_half = int(nchan/2)
        # nchan_half_idx = np.arange(nchan_half)
        # nchan_half_idx_exp = np.exp(2j*np.pi*nchan_half_idx/nchan)
        down_sample_filter_elem = int(self._pfb_filter_coef.shape[0] / nchan)
        filter_idx = np.arange(self._pfb_filter_coef.shape[0]).reshape((down_sample_filter_elem, nchan))
        filter_coeff_2d = self._pfb_filter_coef[filter_idx]

        nu = self.oversampling_factor.numerator
        de = self.oversampling_factor.denominator

        while True:
            x = (yield)
            # t0 = time.time()
            # using precomputed filter is much faster than iterating
            self._pfb_output_mask[ipol,:] = np.sum(
                self._pfb_input_mask[ipol,:][filter_idx] * filter_coeff_2d, axis=0)

            # self.logger.debug(f"_pfb: Applying mask took {time.time()-t0:.6f} seconds")
            # print("output mask: {}".format(self._pfb_output_mask[ipol,:]))
            # shift input mask over by nchan samples
            self._pfb_input_mask[ipol,nchan_norm:] = self._pfb_input_mask[ipol,:-nchan_norm]
            # assign the first nchan samples to the flipped input
            self._pfb_input_mask[ipol,:nchan_norm] = x[::-1]
            # print("input mask: {}".format(self._pfb_input_mask[ipol,:]))

            if self.oversampled:
                if n_os_ctrl == 0:
                    output_mask = self._pfb_output_mask[ipol,:]
                else:
                    shift_idx = (nu-n_os_ctrl)*nchan_overlap
                    output_mask = np.append(
                        self._pfb_output_mask[ipol,shift_idx:],
                        self._pfb_output_mask[ipol,:shift_idx]
                    )
                n_os_ctrl = n_os_ctrl % nu
                n_os_ctrl += 1
            else:
                output_mask = self._pfb_output_mask[ipol,:]
            out = 2*nchan*nchan_half*np.fft.ifft(output_mask)
            print(output_mask)
            print(output_mask.dtype)
            print(out.dtype)
            input(">> ")
            # print(out)
            # input(">> ")
            # fwd_fft = np.fft.fft(output_mask)
            # print(f"out: {out}")
            # print(f"fwd_fft: {fwd_fft}")
            # print(out / fwd_fft)
            # print(2*nchan*nchan_half)
            # print(f"output_mask: {output_mask}")
            # print(f"out: {out}")

            if sink is not None:
                sink.send(out)

            n_iter += 1

            # self.logger.debug(f"_pfb: total loop time: {time.time()-t0:.6f} seconds")

    # @coroutine
    # def _pfb(self, ipol, sink=None, experimental=False):
    #     """
    #     Assumes real input data
    #     """
    #     n_iter = 0
    #     nchan = self._output_nchan
    #     nchan_half = int(nchan/2)
    #     nchan_half_idx = np.arange(nchan_half)
    #     nchan_half_idx_exp = np.exp(2j*np.pi*nchan_half_idx/nchan)
    #     down_sample_filter_elem = int(self._pfb_filter_coef.shape[0] / nchan)
    #     # filter_idx = nchan * np.arange(down_sample_filter_elem)
    #     # filter_idx = np.repeat(filter_idx, nchan).reshape((down_sample_filter_elem, nchan))
    #     # filter_idx += np.arange(nchan)
    #     filter_idx = np.arange(self._pfb_filter_coef.shape[0]).reshape((down_sample_filter_elem, nchan))
    #     filter_coeff_2d = self._pfb_filter_coef[filter_idx]
    #     # _pfb_input_mask is (2, 155)
    #     # _pfb_output_mask is (nchan, 2)
    #     while True:
    #         x = (yield)
    #         # t0 = time.time()
    #         # using precomputed filter
    #         self._pfb_output_mask[ipol,:] = np.sum(
    #             self._pfb_input_mask[ipol,:][filter_idx] * filter_coeff_2d, axis=0)
    #         # for c in range(self._output_nchan):
    #         #     self._pfb_output_mask[ipol, c] = np.sum(
    #         #         self._pfb_input_mask[ipol, c::nchan] * self._pfb_filter_coef[c::nchan]
    #         #     )
    #
    #         # self.logger.debug(f"_pfb: Applying mask took {time.time()-t0:.6f} seconds")
    #         # print("output mask: {}".format(self._pfb_output_mask[ipol,:]))
    #         # shift input mask over by nchan samples
    #         self._pfb_input_mask[ipol,nchan:] = self._pfb_input_mask[ipol,:-nchan]
    #         # assign the first nchan samples to the flipped input
    #         self._pfb_input_mask[ipol,:nchan] = x[::-1]
    #         # print("input mask: {}".format(self._pfb_input_mask[ipol,:]))
    #
    #         # Make complex array by interweaving output mask
    #         out = 2*nchan*nchan_half*np.fft.ifft(self._pfb_output_mask[ipol,:])
    #         # c_out = self._pfb_output_mask[ipol,::2] + 1j*self._pfb_output_mask[ipol,1::2]
    #         # # print("c_out: {}".format(c_out))
    #         # # Do inverse FFT of complex output mask
    #         # ifft_c_out = nchan * nchan_half * np.fft.ifft(c_out)
    #         # # print("ifft_c_out: {}".format(ifft_c_out))
    #         #
    #         # out = np.zeros(nchan, dtype=self.complex_dtype)
    #         #
    #         # t0 = time.time()
    #         # flipped_rolled_conj = np.conj(np.roll(ifft_c_out[::-1],1))
    #         #
    #         # out[:nchan_half] = (
    #         #     0.5*((ifft_c_out+flipped_rolled_conj) -
    #         #     1j*nchan_half_idx_exp *
    #         #     (ifft_c_out-flipped_rolled_conj))
    #         # )
    #         #
    #         # first_conj = np.conj(ifft_c_out[0])
    #         #
    #         # out[nchan_half] = 0.5*(
    #         #     (ifft_c_out[0]+first_conj +
    #         #     1j*(ifft_c_out[0]-first_conj))
    #         # )
    #         #
    #         # out[nchan_half+1:] = np.conj((out[1:nchan_half])[::-1])
    #
    #         # r_out = self._pfb_output_mask[ipol,:]
    #         # # print(c_out_real)
    #         # ifft_r_out = np.fft.ifft(r_out)
    #         # print(2*nchan*nchan_half*ifft_r_out)
    #         # print(out)
    #
    #         # out_real = np.fft.ihfft(c_out_real, n=nchan)
    #         # print(len(out_real))
    #         # print(f"out_real: {out_real}")
    #         # print(f"out: {out}")
    #         # self.logger.debug(f"_pfb: rearranging took {time.time()-t0:.6f} seconds")
    #
    #         # print(f"out: {out}")
    #         if sink is not None:
    #             sink.send(out)
    #
    #
    #         n_iter += 1
    #         # self.logger.debug(f"_pfb: total loop time: {time.time()-t0:.6f} seconds")

    def channelize_conv(self, **kwargs):
        t_total = time.time()

        if self.input_data is None:
            self._load_input_data()
        if self.output_header is None:
            self._init_output_header()
        if self.output_data is None:
            self._init_output_data()
        self._load_pfb_config()
        # self._pfb_filter_coef = self._pfb_filter_coef[:155]

        input_samples = self.input_data[:self._input_samples]

        input_samples_per_pol_dim = int(self._input_samples / (self._input_npol * self._input_ndim))
        output_samples_per_pol_dim = int(self._output_samples / (self._output_npol * self._input_ndim))

        if self._input_ndim > 1:
            input_samples = input_samples.reshape((input_samples_per_pol_dim*self._input_npol, self._input_ndim))
            input_samples = input_samples[:,0] + 1j*input_samples[:,1]

        input_samples = input_samples.reshape((input_samples_per_pol_dim, self._input_npol))

        # do any downsampling necessary for conversion from real to complex data.
        if int(self._ndim_ratio) != 1:
            input_samples = input_samples[::int(self._ndim_ratio),:]

        nchan = self._output_nchan

        output_filtered = np.zeros(
            (output_samples_per_pol_dim + 1, nchan), dtype=self.float_dtype)
        for p in range(self._input_npol):
            p_idx = self._output_ndim * p
            t0 = time.time()
            filter(
                input_samples[:,p].copy(),
                self._pfb_filter_coef,
                output_filtered,
                nchan
            )
            self.logger.debug(f"channelize_conv: Call to filter took {time.time()-t0:.4f} seconds")
            # for c in range(nchan):
            #     filter = self._pfb_filter_coef[c::nchan]
            #     if c == 0:
            #         input_chunk = input_samples[::nchan, p]
            #         filtered = scipy.signal.lfilter(filter, [1.0], input_chunk)
            #         # filtered = np.append(filtered, 0)
            #     else:
            #         input_chunk = input_samples[(nchan-c)::nchan, p]
            #         filtered = scipy.signal.lfilter(filter, [1.0], input_chunk)
            #         # filtered = np.insert(filtered, 0, 0)
            #     # input_chunk = input_samples[c::nchan, p]
            #     # filtered = scipy.signal.lfilter(filter, [1.0], input_chunk)
            #     filtered = np.insert(filtered, 0, 0)
            #     filtered = filtered.astype(self.float_dtype)

                # print(f"{time.time() - t0}")
                #
                # fig, axes = plt.subplots(4, 1)
                # for ax in axes:
                #     ax.grid(True)
                # axes[0].plot(filtered_c)
                # axes[1].plot(filtered)
                # axes[2].plot(scipy.signal.fftconvolve(filtered_c, filtered[::-1], mode="same"))
                # axes[3].plot((filtered_c - filtered)**2)
                # plt.show()
                #
                # print(filtered_c.shape)
                # print(filtered.shape)
                # print(input_chunk.shape)
                # print(np.allclose(filtered_c, filtered))
                # output_filtered[:,c] = filtered

                # ifft_filtered = nchan**2 * np.fft.ifft(filtered)
                # print(np.mean(np.real(filtered)))
                # print(np.mean(np.imag(filtered)))
                # filter = filter_coef_padded[c::nchan]
                # filtered = np.fft.ifft(np.fft.fft(filter) * np.fft.fft(input_chunk))
                # self.output_data[:,c,p_idx] = np.real(ifft_filtered)
                # self.output_data[:,c,p_idx+1] = np.imag(ifft_filtered)
            # for j in range(output_samples_per_pol_dim):
            #     print(output_filtered[j,:])
            #     input(">> ")
            output_filtered_fft = nchan**2 * np.fft.ifft(output_filtered, n=nchan, axis=1)[:-1,:]
            # for j in range(output_samples_per_pol_dim):
            #     print(output_filtered_fft[j,:])
            #     input(">> ")
            self.output_data[:,:,p_idx] = np.real(output_filtered_fft)
            self.output_data[:,:,p_idx+1] = np.imag(output_filtered_fft)


        split = self.output_file_path.split(".")
        split.insert(1, "conv")
        self.output_file_path = ".".join(split)
        self._dump_data(self.output_header, self.output_data, **kwargs)
        self.logger.debug(f"channelize_conv: Took {time.time() - t_total:.4f} seconds to channelize")


    def channelize(self, diagnostic_plots=False, **kwargs):

        t_total = time.time()

        if self.input_data is None:
            self._load_input_data()
        if self.output_header is None:
            self._init_output_header()
        if self.output_data is None:
            self._init_output_data()
        self._load_pfb_config(diagnostic_plots=diagnostic_plots)

        if diagnostic_plots:
            plt.ion()
            fig_input, axes_input = plt.subplots(2, 2)
            fig_output, axes_output = plt.subplots(self._output_nchan, int(self._input_npol*self._output_ndim))
            # fig_input.tight_layout()
            # fig_output.tight_layout()

        norm_chan = self.oversampling_factor.normalize(self._output_nchan)

        # output_chunk = np.zeros((self._output_samples, self._output_nchan, self._output_npol), dtype=self.complex_dtype)

        pfb_consumer = [Consumer() for i in range(2)]
        pfb_coro = [self._pfb(i, pfb_consumer[i]) for i in range(2)]

        input_samples = self.input_data[:self._input_samples]

        input_samples_per_pol_dim = int(self._input_samples / (self._input_npol * self._input_ndim))
        output_samples_per_pol_dim = int(self._output_samples / (self._output_npol * self._input_ndim))

        self.logger.debug(f"channelize: input_samples_per_pol_dim: {input_samples_per_pol_dim}")
        self.logger.debug(f"channelize: output_samples_per_pol_dim: {output_samples_per_pol_dim}")

        if self._input_ndim > 1:
            input_samples = input_samples.reshape((input_samples_per_pol_dim*self._input_npol, self._input_ndim))
            input_samples = input_samples[:,0] + 1j*input_samples[:,1]

        input_samples = input_samples.reshape((input_samples_per_pol_dim, self._input_npol))

        # do any downsampling necessary for conversion from real to complex data.
        if int(self._ndim_ratio) != 1:
            input_samples = input_samples[::int(self._ndim_ratio),:]
        p_idx = None
        chunk_size = 5*int(1e4)
        for p in range(self._input_npol):
            tpol = time.time()
            t0 = time.time()
            coro = pfb_coro[p]
            sink = pfb_consumer[p]
            p_idx = self._output_ndim * p
            for j in range(output_samples_per_pol_dim):
                coro.send(input_samples[norm_chan*j:norm_chan*(j+1),p])
                self.output_data[j,:,p_idx] = np.real(sink.val)
                self.output_data[j,:,p_idx + 1] = np.imag(sink.val)
                if (j % chunk_size) == 0 and j > 0:
                    self.logger.debug(
                        (f"channelize: processing samples "
                         f"{j - chunk_size} - {j} / {output_samples_per_pol_dim} "
                         f"samples took {time.time()-t0:.4f} seconds"))
                    t0 = time.time()

            self.logger.debug(f"channelize: pol {p} took {time.time()-tpol:.4f} seconds")
        # for i in range(self._n_series):
        #     t0 = time.time()
        #     input_increment = int(self._input_npol * self._input_ndim * self._input_samples)
        #     input_chunk = self.input_data[i*input_increment:(i+1)*input_increment]
        #
        #     if self._input_ndim == 2:
        #         input_chunk = input_chunk.reshape((self._input_samples*self._input_npol, self._input_ndim))
        #         input_chunk = input_chunk[:,0] + 1j*input_chunk[:,1]
        #
        #     input_chunk = input_chunk.reshape((self._input_samples, self._input_npol))
        #
        #     # do any downsampling necessary for conversion from real to complex data.
        #     if int(self._ndim_ratio) != 1:
        #         input_chunk = input_chunk[::int(self._ndim_ratio),:]
        #
        #
        #     for p in range(self._input_npol):
        #         coro = pfb_coro[p]
        #         sink = pfb_consumer[p]
        #         for j in range(self._output_samples):
        #             coro.send(input_chunk[norm_chan*j:norm_chan*(j+1),p])
        #             output_chunk[j,:,p] = sink.val.copy()
        #             # input(">> ")
        #
        #     s = self._output_samples*i
        #     e = self._output_samples*(i+1)
        #     self.output_data[s:e,:,0] = np.real(output_chunk[:,:,0])
        #     self.output_data[s:e,:,1] = np.imag(output_chunk[:,:,0])
        #     self.output_data[s:e,:,2] = np.real(output_chunk[:,:,1])
        #     self.output_data[s:e,:,3] = np.imag(output_chunk[:,:,1])
        #     self.logger.debug(f"channelize: Loop {i}/{self._n_series} took {time.time() - t0:.4f} seconds")
        #
        #     if diagnostic_plots:
        #         axes_input[0,0].plot(np.real(input_chunk[:, 0]))
        #         axes_input[0,0].set_title("Pol 1 Real")
        #         axes_input[1,0].plot(np.imag(input_chunk[:, 0]))
        #         axes_input[1,0].set_title("Pol 1 Imag")
        #         axes_input[0,1].plot(np.real(input_chunk[:, 1]))
        #         axes_input[0,1].set_title("Pol 2 Real")
        #         axes_input[1,1].plot(np.imag(input_chunk[:, 1]))
        #         axes_input[1,1].set_title("Pol 2 Imag")
        #
        #         for p in range(self._input_npol):
        #             for c in range(self._output_nchan):
        #                 for z in range(self._output_ndim):
        #                     z_name = "Real"
        #                     if z == 1:
        #                         z_name = "Imag"
        #                     idx = z + p*self._output_ndim
        #                     ax = axes_output[c, idx]
        #                     plt_data = self.output_data[idx,c,s:e]
        #                     ax.plot(np.arange(len(plt_data)), plt_data)
        #                     ax.grid(True)
        #                     ax.set_title(f"Output {z_name}, pol {p}, channel {c}")
        #                     for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        #                         item.set_fontsize(5)
        #         input(">> ")
        #


        self._dump_data(self.output_header, self.output_data, **kwargs)
        self.logger.debug(f"channelize: Took {time.time() - t_total:.4f} seconds to channelize")

def compare_matlab_py(*fnames, **kwargs):
    comp_dat = []
    min_size = -1
    for fname in fnames:
        with open(fname, "rb") as input_file:
            buffer = input_file.read()
            header = np.frombuffer(buffer, dtype='c', count=PFBChannelizer.header_size)
            data = np.frombuffer(buffer, dtype=PFBChannelizer.float_dtype, offset=PFBChannelizer.header_size)
            if min_size == -1:
                min_size = data.shape[0]
            elif data.shape[0] < min_size:
                min_size = data.shape[0]
            comp_dat.append(data)
    comp_dat = [d[:min_size] for d in comp_dat]
    fig, axes = plt.subplots(len(fnames)+2)
    for ax in axes:
        ax.grid(True)
    axes[0].plot(comp_dat[0])
    axes[1].plot(comp_dat[1])
    diff_squared = (comp_dat[0] - comp_dat[1])**2
    axes[2].plot(diff_squared)
    axes[3].plot(scipy.signal.fftconvolve(comp_dat[0], comp_dat[1][::-1], mode="same"))
    argmax = np.argmax(diff_squared)
    if not np.allclose(*comp_dat, **kwargs):
        print(comp_dat[0][argmax], comp_dat[1][argmax])
        # for i in range(3):
        #     axes[i].set_xlim([argmax-100, argmax+100])

        print(f"{np.sum(diff_squared == 0.0)}/{diff_squared.shape[0]} are zero.")
    else:
        print("all close!")

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    # input_file_name = "simulated_pulsar.noise_0.0.nseries_1.ndim_1.dump"
    input_file_name = "simulated_pulsar.noise_0.0.nseries_5.ndim_1.dump"
    # input_file_name = "simulated_pulsar.noise_0.0.nseries_5.ndim_2.dump"
    input_file_path = os.path.join(data_dir, input_file_name)
    # os = Rational(8,7)
    os = Rational(1,1)
    channelizer = PFBChannelizer(
        input_file_path, os
    )
    # channelizer._load_input_data()
    # channelizer._init_output_data()
    # channelizer._init_output_header()
    # channelizer._dump_data(channelizer.output_header, channelizer.output_data)
    # channelizer.channelize()
    channelizer.channelize_conv()

    compare_matlab_py(
        "data/py_channelized.conv.noise_0.0.nseries_5.ndim_1.cs.dump",
        # "data/py_channelized.noise_0.0.nseries_5.ndim_1.cs.dump",
        "data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_1.cs.dump",
        rtol=1e-6,
        atol=1e-6
    )
