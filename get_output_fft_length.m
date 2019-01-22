function output_fft_length = get_output_fft_length (input_fft_length, n_chan, OS_Nu, OS_De)
  output_fft_length = input_fft_length * (OS_De/OS_Nu) * n_chan;
end
