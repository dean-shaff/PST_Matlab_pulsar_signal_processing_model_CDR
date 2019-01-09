function compare_inversion(simulated_pulsar_filename, inverted_file_name)
  % compare simulated pulsar raw data to inverted data.

  % read in input data
  hdr_map = read_header(simulated_pulsar_filename, containers.Map());
  hdr_size = str2num(hdr_map('HDR_SIZE'));
  n_dim = str2num(hdr_map('NDIM'));

  fid_in = fopen(simulated_pulsar_filename);
  % skip header
  fread(fid_in, hdr_size, 'uint8');

  data_simulated = fread(fid_in, 'single');
  data_simulated = reshape(data_simulated, 1, []);
  fclose(fid_in);

  if n_dim == 2
    fprintf('compare_inversion: pulsar data is complex');
    data_simulated = reshape(data_simulated, n_dim, []);
    data_simulated = complex(data_simulated(1, :), data_simulated(2, :));
  else
    fprintf('compare_inversion: pulsar data is real');
  end

  % read in inverted data (saved in .mat matlab file)
  load(inverted_file_name);
  data_inverted = inverted;

  size(data_simulated)
  size(data_inverted)


  len = length(data_inverted);

  subplot(311); plot((1:len), real(data_simulated(1,1:len)), (1:len), real(data_inverted(1:len))); box on; grid on;
  legend({'original Real', 'inverted Real'}); title('original Real vs inverted Real');
  subplot(312); plot((1:len), imag(data_simulated(1,1:len)), (1:len), imag(data_inverted(1:len))); box on; grid on;
  legend({'original Imag', 'inverted Imag'}); title('original Imag vs inverted Imag'); xlabel('time');

  cross_corr_real = xcorr(real(data_simulated(1, 1:len)), real(data_inverted(1:len)));
  cross_corr_imag = xcorr(imag(data_simulated(1, 1:len)), imag(data_inverted(1:len)));
  len_cross_corr = length(cross_corr_real);

  subplot(313); plot((1:len_cross_corr), cross_corr_real, (1:len_cross_corr), cross_corr_imag); box on; grid on;
  legend({'xcorr Real', 'xcorr Imag'}); title('Vin vs data_inverted cross correlation');

  % Time domain comparison with original input - good for integer sample
  % delays when those delays have been sync'd out
  % centre_data_simulated = Nin/4;
  % centre_data_inverted = centre_data_simulated + compare_offset;
  %
  % plot_range = 25;
  % figure; set(gcf,'Visible', 'off');
  %
  % t_plot = (-plot_range+1:plot_range);
  % data_inverted_plot = data_inverted(centre_data_inverted-plot_range+1:centre_data_inverted+plot_range);
  % data_simulated_plot = data_simulated(1,centre_data_simulated-plot_range+1:centre_data_simulated+plot_range);
  %
  % subplot(311); plot(t_plot,...
  %   real(data_inverted_plot),...
  %   t_plot,...
  %   real(data_simulated_plot));
  % box on; grid on; title('data_inverted vs data_simulated Real'); legend({'data_inverted Real', 'data_simulated Real'});
  % subplot(312); plot(t_plot,...
  %   imag(data_inverted_plot),...
  %   t_plot,...
  %   imag(data_simulated_plot));
  % box on; grid on; title('data_inverted vs data_simulated Imag'); legend({'data_inverted Imag', 'data_simulated Imag'});
  % xlabel('time');
  %
  % xcorr_data_inverted_data_simulated = xcorr(data_inverted_plot, data_simulated_plot);
  % subplot(313); plot((1:length(xcorr_data_inverted_data_simulated)), real(xcorr_data_inverted_data_simulated), (1:length(xcorr_data_inverted_data_simulated)), imag(xcorr_data_inverted_data_simulated));
  % box on; grid on; title('Cross correlation data_inverted vs data_simulated');
  % % calculate total RMS error over Nsamp samples
  % Nsamp = 200;
  % Realerr = 0;
  % Imagerr = 0;
  %
  % for j = centre_data_simulated-(Nsamp/2):centre_data_simulated+(Nsamp/2-1),
  %     Realerr = Realerr + (real(data_simulated(1,j)) - real(data_inverted(j+compare_offset)))^2;
  %     Imagerr = Imagerr + (imag(data_simulated(1,j)) - imag(data_inverted(j+compare_offset)))^2;
  % end;
  %
  % RMSerr_real = (Realerr/Nsamp)^0.5
  % RMSerr_imag = (Imagerr/Nsamp)^0.5
  %
  %
  %
  % % Frequency domain comparison with original input - good for fractional
  % % sample delays to avoid the need for interpolation of the input time series
  % data_simulated_shift = data_simulated(1,1-compare_offset:len-compare_offset);
  % % data_simulated = fft(data_simulated_shift);
  % fft_data_simulated = fft(data_simulated(1:len));
  %
  % figure;
  % subplot(211); plot((1:len),abs(fft_data_simulated), (1:len), abs(FFFF));
  %   xlim([-100, len]);
  %   set(gca, 'YScale', 'log');
  %   legend({'Mag FFT data_simulated', 'Mag FFFF'});
  %   box on; grid on; title('FFT data_simulated vs FFFF magnitude');
  % subplot(212); plot((1:len),angle(fft_data_simulated), (1:len), angle(FFFF));
  %   legend({'Phase FFT data_simulated', 'Phase FFFF'});
  %   box on; grid on; title('FFT data_simulated vs FFFF phase'); %xlabel('time');

  % cross-power spectrum of FFFF and a similar length data_simulated
  % CP = FFFF.*transpose(conj(data_simulated));
  % figure;%set(gcf,'Visible', 'off');
  % subplot(211); plot((1:len),abs(CP),'Linewidth',2);
  % box on; grid on; title('Cross-Power Mag'); set(gca, 'YScale', 'log')
  % subplot(212); plot((1:len),angle(CP),'Linewidth',2);
  % box on; grid on; title('Cross-Power Phase'); xlabel('time');

  % sum the total cross-power magnitude
  % total_cp = sum(abs(CP))


end % end compare_inversion
