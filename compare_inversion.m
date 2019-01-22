function compare_inversion(simulated_pulsar_filename, inverted_filename, offset_, plot_time_, plot_freq_)
  % compare simulated pulsar raw data to inverted data.
  close all;
  offset = 0;
  if exist('offset_', 'var')
    offset = offset_;
  end

  plot_time = 0;
  if exist('plot_time_', 'var')
    plot_time = plot_time_;
  end

  plot_freq = 0;
  if exist('plot_freq_', 'var')
    plot_freq = plot_freq_;
  end


  [data_simulated, hdr_map] = load_simulated_pulsar_dump(simulated_pulsar_filename);
  n_pol = str2num(hdr_map('NPOL'));
  OS_Nu = 1;
  OS_De = 1;
  % OS_Nu = 8;
  % OS_De = 7;
  % [OS_Nu, OS_De] = get_os_factor_from_hdr_map(hdr_map);
  output_fft_length = get_output_fft_length(32768, 8, OS_Nu, OS_De);
  % data_simulated = reshape(data_simulated, [], n_pol);
  % data_simulated = transpose(reshape(data_simulated, n_pol, []));

  if endsWith(inverted_filename, '.mat')
    % read in inverted data (saved in .mat matlab file)
    fprintf('Using .mat file\n');
    load(inverted_filename);
    data_inverted = squeeze(inverted);
    data_inverted = transpose(data_inverted);

  elseif endsWith(inverted_filename, '.dump')
    data_inverted = load_dspsr_dump(inverted_filename);
  end

  data_inverted = data_inverted ./ output_fft_length; % normalized inverse FFT

  % data_inverted = reshape(data_inverted, [], n_pol);

  % data_inverted = data_inverted ./ max(data_inverted);
  % data_simulated = data_simulated ./ max(data_simulated);
  fprintf('max of data_inverted: %f\n', max(data_inverted));
  fprintf('length of data_inverted: %d\n', length(data_inverted));
  fprintf('length of data_simulated: %d\n', length(data_simulated));

  percentage = 1.0;
  len = length(data_inverted);
  len = round(percentage*len);

  if plot_time
    figure;
    set(gcf, 'Position', [10, 10, 1210, 810]);
    for pol=1:n_pol
      x = 1:len;
      incr = pol - 1;
      real_sim = real(data_simulated(1:len, pol));
      imag_sim = imag(data_simulated(1:len, pol));
      real_inv = real(data_inverted(1:len, pol));
      imag_inv = imag(data_inverted(1:len, pol));

      subplot(3, n_pol, incr + 1);
        plot(x, real_sim, x, real_inv);
        % patchline(x, real_sim, 'linestyle','--','edgecolor','r','linewidth',3,'edgealpha',0.2);
        % patchline(x, real_inv, 'linestyle','--','edgecolor','b','linewidth',3,'edgealpha',0.2);
        box on; grid on;
        legend({'Original Real', 'Inverted Real'});
        title(sprintf('Pol %d Original vs Inverted Real', pol));

      subplot(3, n_pol, incr + 3);
        plot(x, imag_sim, x, imag_inv);
        box on; grid on;
        legend({'Original Imag', 'Inverted Imag'});
        title(sprintf('Pol %d Original vs Inverted Imag', pol)); xlabel('time');

      cross_corr_real = xcorr(...
        real_sim, real_inv);
      cross_corr_imag = xcorr(...
        imag_sim, imag_inv);
      [argvalue, argmax] = max(cross_corr_real);
      len_cross_corr = length(cross_corr_real);
      offset_from_middle = argmax - round(len_cross_corr/2);
      fprintf('argmax of cross correlation is %d, offset %d from middle \n', argmax, offset_from_middle);
      x_x_corr = 1:len_cross_corr;
      subplot(3, n_pol, incr + 5);
      plot(x_x_corr, cross_corr_real, x_x_corr, cross_corr_imag);
      box on; grid on;
      legend({'xcorr Real', 'xcorr Imag'});
      title(sprintf('Pol %d Original vs Inverted cross correlation', pol));
    end
    fprintf('Done plotting time domain comparison\n');
  end
  % sgtitle('Original vs Inverted time domain comparison');



  % Time domain comparison with original input - good for integer sample
  % delays when those delays have been sync'd out
  % centre_data_simulated = Nin/4;
  % centre_data_inverted = centre_data_simulated + offset;
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
  %     Realerr = Realerr + (real(data_simulated(1,j)) - real(data_inverted(j+offset)))^2;
  %     Imagerr = Imagerr + (imag(data_simulated(1,j)) - imag(data_inverted(j+offset)))^2;
  % end;
  %
  % RMSerr_real = (Realerr/Nsamp)^0.5
  % RMSerr_imag = (Imagerr/Nsamp)^0.5
  %
  %
  %
  % Frequency domain comparison with original input - good for fractional
  % sample delays to avoid the need for interpolation of the input time series

  if plot_freq
    figure;
    set(gcf, 'Position', [10, 10, 1210, 810]);
    len = round(0.2*len);
    for pol=1:n_pol
      sim_pol = data_simulated(1-offset:len-offset, pol);
      inv_pol = data_inverted(1-offset:len-offset, pol);

      fft_sim_pol = fft(sim_pol);
      fft_inv_pol = fft(inv_pol);
      x = 1:len;
      incr = pol - 1;

      subplot(3, 2, 1 + incr);
        plot(x, abs(fft_sim_pol), x, abs(fft_inv_pol));
        xlim([-100, len]);
        legend({'Magnitude of FFT of Original', 'Magnitude of FFT of Inverted'});
        box on; grid on;
        title(sprintf('Pol: %d: Magnitude of FFT of Original vs Inverted', pol));

      subplot(3, 2, 3 + incr);
        plot(x, mag2db(abs(fft_sim_pol)), x, mag2db(abs(fft_inv_pol)));
        legend({'Log 20 Magnitude of FFT of Original', 'Log 20 Magnitude of FFT of Inverted'});
        box on; grid on;
        title(sprintf('Pol: %d: Log 20 Magnitude of FFT of Original vs Inverted', pol));

      cross_power = fft_inv_pol.*conj(fft_sim_pol);

      subplot(3, 2, 5 + incr); plot(x, abs(cross_power));
        legend({'Magnitude of Cross power'});
        box on; grid on;
        title(sprintf('Pol: %d: Cross Power', pol));


    end
    fprintf('Done plotting frequency domain comparison\n')
  end
  % sgtitle('Original vs Inverted Frequency domain comparison')
  % cross-power spectrum of FFFF and a similar length data_simulated
  % CP = FFFF.*transpose(conj(data_simulated));
  % figure;  % set(gcf,'Visible', 'off');
  % subplot(211); plot((1:len),abs(CP),'Linewidth',2);
  % box on; grid on; title('Cross-Power Mag'); set(gca, 'YScale', 'log')
  % subplot(212); plot((1:len),angle(CP),'Linewidth',2);
  % box on; grid on; title('Cross-Power Phase'); xlabel('time');

  % sum the total cross-power magnitude
  % total_cp = sum(abs(CP))


end % end compare_inversion
