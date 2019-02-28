function inverted_filename = PFBinversion (filename_in, verbose_, method_)

  % PFB Inversion
  % Takes data that has been channelized with polyphase filterbank and inverts transform.
  % adapted from Ian Morrison's code in PFB_inversion_by_FFT_complex_input repository.
  % Dean Shaff 2018

  % filename_in = file name corresponding to a file that contains
  %   data that has been channelized with PFBchannelizer. Data file should
  %   contain 4096 byte DADA header. The structure of the data file should be
  %   as follows: (n_pol, n_dim, n_chan, n_samples), where n_pol is the number
  %   of polarisations, n_dim is either 1 (for real data) or 2 (for complex data)
  %   n_chan is the number of channels created by PFBchannelizer code, and
  %   n_samples is the number of samples in each channel.
  % verbose_ = true for verbose output, false for quiet output.


  verbose = 0;
  if exist('verbose_', 'var')
    verbose = verbose_;
  end


  method = 0;
  if exist('method_', 'var')
    method = method_;
  end


  % read in input data
  hdr_map = read_header(filename_in, containers.Map());

  hdr_size = str2num(hdr_map('HDR_SIZE'));

  n_chan = str2num(hdr_map('NCHAN'));
  n_pol = str2num(hdr_map('NPOL'));
  n_dim = str2num(hdr_map('NDIM'));

  OS_factor = hdr_map('OS_FACTOR');
  OS_factor_split = strsplit(OS_factor, '/');
  OS_Nu = str2num(OS_factor_split{1});
  OS_De = str2num(OS_factor_split{2});

  fid_in = fopen(filename_in);
  fread(fid_in, hdr_size, 'uint8'); % skip the header

  data_in = fread(fid_in, 'single');
  fclose(fid_in);

  if verbose
    fprintf('OS_Nu: %d\n', OS_Nu);
    fprintf('OS_De: %d\n', OS_De);
  end

  data_in = reshape(data_in, [], 1);
  data_in = reshape(data_in, n_pol*n_dim, n_chan, []);

  if n_dim == 2
    size_data_in = size(data_in);
    dat_temp = zeros(n_pol, n_chan, size_data_in(end));
    dat_temp(1,:,:) = squeeze(complex(data_in(1,:,:), data_in(2,:,:)));
    dat_temp(2,:,:) = squeeze(complex(data_in(3,:,:), data_in(4,:,:)));
    data_in = dat_temp;
  else
    throw(MException('No support for real input data'));
  end
  size_data_in = size(data_in);
  n_samples = size_data_in(end);
  input_offset = 1;
  data_in = data_in(:, :, input_offset:end);

  if method == 1
    os_keep_region = round((OS_De/OS_Nu)*n_samples);
    equalise_ripple = 1;
    fine_channels = zeros(n_pol, n_chan, os_keep_region);

    % process fine channels
    for chan=1:n_chan
      for pol=1:n_pol
        fprintf('Processing fine channel %d, polarization %d\n', chan, pol);
         fine_chan_proc_res = fine_chan_proc(...
          data_in(pol, chan, :), n_samples, chan, OS_Nu, OS_De, equalise_ripple, 1);
        % fine_chan_proc_res = fine_chan_proc(...
        %  data_in(pol, 1, chan, offset:end), n_samples, chan, OS_Nu, OS_De, equalise_ripple, 1);

        fine_channels(pol, chan, :) = reshape(fine_chan_proc_res, 1, []);
      end
    end

    inverted = zeros(n_pol, os_keep_region*n_chan);

    % invert data
    for pol=1:n_pol
      fprintf('Inverting polarization %d\n', pol);
      fine_channel_pol = reshape(fine_channels(pol, :, :), n_chan, os_keep_region);
      invert_res = invert(...
        fine_channel_pol, os_keep_region, n_chan, OS_Nu, OS_De, 0, 0, 1);
      inverted(pol, :) = invert_res;
    end
  elseif method == 2
    inverted = dspsr_inversion(squeeze(data_in), OS_Nu, OS_De, 1);
  elseif method == 3
    inverted = inversion(squeeze(data_in), OS_Nu, OS_De, 1);
  end
  % save inverted data
  inverted_filename = strcat(filename_in, sprintf('.inverted.%s.mat', method));
  save(inverted_filename, 'inverted')

end % end of PFBinversion

function inverted=inversion(data_in, OS_Nu, OS_De, verbose_)
  verbose=0;
  if exist('verbose_', 'var')
    verbose = verbose_;
  end

  if verbose
    size(data_in)
    fprintf('inversion: OS_Nu: %d\n', OS_Nu);
    fprintf('inversion: OS_De: %d\n', OS_De);
  end
  [n_pol, n_chan, n_dat] = size(data_in);
  input_fft_length = 32768;  % size of forward FFT, from dspsr logs
  % output_fft_length = 229376;  % size of backward FFT, from dspsr logs
  % input_fft_length = 737280;
  output_fft_length = input_fft_length * (OS_De/OS_Nu) * n_chan;
  % input_discard = struct('pos', 960,'neg', 1040); % input dedispersion removal region, from dspsr logs
  input_discard = struct('pos', 835,'neg', 909); % input dedispersion removal region, from dspsr logs
  input_discard_total = input_discard.pos + input_discard.neg;
  input_sample_step = input_fft_length - input_discard_total;
  % output_discard = struct('pos', 6720,'neg', 7280);  % output dedispersion removal region, from dspsr logs
  output_discard = struct('pos', 6680,'neg', 7272);  % output dedispersion removal region, from dspsr logs
  output_discard_total = output_discard.pos + output_discard.neg;
  output_sample_step = output_fft_length - output_discard_total;

  input_n_parts = round((n_dat - input_discard_total) / input_sample_step);
  if input_n_parts == 1
    input_n_parts = input_n_parts + 2
  end
  output_n_parts = round((n_dat - output_discard_total) / output_sample_step);

  input_os_keep = (OS_De/OS_Nu) * input_fft_length;
  input_os_discard = (input_fft_length - input_os_keep) / 2;

  inverted = zeros(1, n_pol, output_n_parts * output_sample_step);

  if verbose
    fprintf('max(data_in): %f\n', max(data_in(:)));
    fprintf('n_pol: %d\n', n_pol);
    fprintf('n_chan: %d\n', n_chan);
    fprintf('n_dat: %d\n', n_dat);
    fprintf('input_fft_length: %d\n', input_fft_length);
    fprintf('output_fft_length: %d\n', output_fft_length);
    fprintf('input_sample_step: %d\n', input_sample_step);
    fprintf('output_sample_step: %d\n', output_sample_step);
    fprintf('input_discard_total: %d\n', input_discard_total);
    fprintf('output_discard_total: %d\n', output_discard_total);
    fprintf('input_n_parts: %d\n', input_n_parts);
    fprintf('output_n_parts: %d\n', output_n_parts);
    fprintf('input_os_discard: %d\n', input_os_discard);
    fprintf('input_os_keep: %d\n', input_os_keep);
  end
  max_fft = 0;
  for i_part=1:input_n_parts-2
    if verbose
      fprintf('Starting loop %d/%d\n', i_part, input_n_parts);
    end

    for i_pol=1:n_pol
      stitched_freq = zeros(1, n_chan*input_os_keep);

      for i_chan=1:n_chan
        if verbose
          % fprintf('loop %d/%d: pol: %d, chan: %d\n', i_part, input_n_parts, i_pol, i_chan);
        end
        % idx = (i_part-1) * (input_sample_step);
        idx = (i_part-1) * input_fft_length;
        time_chan_pol = data_in(i_pol, i_chan, idx+1:idx+input_fft_length);
        % following the scheme for assembling channelized spectra from Ian Morrison's PFB inversion code.
        freq_chan_pol = fft(time_chan_pol, input_fft_length);
        % the following discards high frequency components.
        % freq_chan_pol_keep = freq_chan_pol(input_os_discard+1:input_os_keep+input_os_discard);
        % we have to stitch negative and positive frequency components differently.
        n_2 = input_os_keep / 2;
        if i_chan == 1
          pos_start = 1;
          pos_end = n_2;
          neg_start = output_fft_length - n_2 + 1;
          neg_end = output_fft_length;
        else
          neg_start = (i_chan-2)*input_os_keep + n_2 + 1;
          neg_end = neg_start + n_2 - 1;
          pos_start = neg_end + 1;
          pos_end = pos_start + n_2 - 1;
        end
        fprintf('chan: %d, pos_start: %d, pos_end: %d, neg_start: %d, neg_end: %d\n', i_chan, pos_start, pos_end, neg_start, neg_end);
        stitched_freq(1, pos_start:pos_end) = squeeze(freq_chan_pol(1, 1, 1:n_2));
        stitched_freq(1, neg_start:neg_end) = squeeze(freq_chan_pol(1, 1, (input_fft_length - n_2)+1:end));

        % fprintf('pos_start: %d, pos_end: %d, neg_start: %d, neg_end: %d\n', pos_start, pos_end, neg_start, neg_end);
        % size(stitched_freq(1, pos_start:pos_end))
        % size(stitched_freq(1, neg_start:neg_end))
        % size(squeeze(freq_chan_pol_keep(1, 1, n_2+1:end)))
        % size(squeeze(freq_chan_pol_keep(1, 1, 1:n_2)))

        % stitched_freq(1, (i_chan-1)*input_os_keep+1:(i_chan)*input_os_keep) = squeeze(freq_chan_pol_keep(1, 1, :));
      end
      if max(stitched_freq(:)) > max_fft
        max_fft = max(stitched_freq(:));
      end
      stitched_freq = ifftshift(stitched_freq);

      % FFTW doesnt do normalization
      % stitched_time = ifft(stitched_freq, output_fft_length) .* output_fft_length;
      % stitched_time_keep = stitched_time(1, output_discard.pos+1:end-output_discard.neg);
      % % stitched_time_keep = stitched_time(1, 1:output_fft_length-output_discard_total);
      % idx = (i_part-1) * output_sample_step;
      % inverted(1, i_pol, idx+1:idx+output_sample_step) = stitched_time_keep;

      stitched_time = ifft(stitched_freq, output_fft_length) .* output_fft_length;
      stitched_time_keep = stitched_time;
      idx = (i_part-1) * output_fft_length;
      inverted(1, i_pol, idx+1:idx+output_fft_length) = stitched_time_keep;
    end
  end
  % fprintf('max(data_in): %f\n', max(data_in(:)));
  % fprintf('max_fft: %f\n', max_fft);
  % fprintf('max(inverted): %f\n', max(inverted(:)));

end  % end function inversion


function inverted=dspsr_inversion(data_in, OS_Nu, OS_De, verbose_)
  % replicate the filterbank inversion used in dspsr.
  verbose=0;
  if exist('verbose_', 'var')
    verbose = verbose_;
  end
  size(data_in)
  [n_pol, n_chan, n_dat] = size(data_in);
  input_fft_length = 32768;  % size of forward FFT, from dspsr logs
  % output_fft_length = 229376;  % size of backward FFT, from dspsr logs
  output_fft_length = input_fft_length * (OS_De/OS_Nu) * n_chan;
  % input_discard = struct('pos', 960,'neg', 1040); % input dedispersion removal region, from dspsr logs
  input_discard = struct('pos', 835,'neg', 909); % input dedispersion removal region, from dspsr logs
  input_discard_total = input_discard.pos + input_discard.neg;
  input_sample_step = input_fft_length - input_discard_total;
  % output_discard = struct('pos', 6720,'neg', 7280);  % output dedispersion removal region, from dspsr logs
  output_discard = struct('pos', 6680,'neg', 7272);  % output dedispersion removal region, from dspsr logs
  output_discard_total = output_discard.pos + output_discard.neg;
  output_sample_step = output_fft_length - output_discard_total;

  input_n_parts = round((n_dat - input_discard_total) / input_sample_step);
  output_n_parts = round((n_dat - output_discard_total) / output_sample_step);

  input_os_keep = (OS_De/OS_Nu) * input_fft_length;
  input_os_discard = (input_fft_length - input_os_keep) / 2;

  inverted = zeros(1, n_pol, output_n_parts * output_sample_step);

  if verbose
    fprintf('max(data_in): %f\n', max(data_in(:)));
    fprintf('n_pol: %d\n', n_pol);
    fprintf('n_chan: %d\n', n_chan);
    fprintf('n_dat: %d\n', n_dat);
    fprintf('input_fft_length: %d\n', input_fft_length);
    fprintf('output_fft_length: %d\n', output_fft_length);
    fprintf('input_sample_step: %d\n', input_sample_step);
    fprintf('output_sample_step: %d\n', output_sample_step);
    fprintf('input_discard_total: %d\n', input_discard_total);
    fprintf('output_discard_total: %d\n', output_discard_total);
    fprintf('input_n_parts: %d\n', input_n_parts);
    fprintf('output_n_parts: %d\n', output_n_parts);
    fprintf('input_os_keep: %d\n', input_os_keep);
  end
  max_fft = 0
  for i_part=1:input_n_parts-2
    if verbose
      fprintf('Starting loop %d/%d\n', i_part, input_n_parts);
    end

    for i_pol=1:n_pol
      stitched_freq = zeros(1, n_chan*input_os_keep);
      for i_chan=1:n_chan
        if verbose
          % fprintf('loop %d/%d: pol: %d, chan: %d\n', i_part, input_n_parts, i_pol, i_chan);
        end
        idx = (i_part-1) * (input_sample_step);
        % idx = (i_part-1) * input_fft_length;
        time_chan_pol = data_in(i_pol, i_chan, idx+1:idx+input_fft_length);
        freq_chan_pol = fft(time_chan_pol, input_fft_length);
        freq_chan_pol_keep = freq_chan_pol(input_os_discard+1: input_os_keep + input_os_discard);
        stitched_freq(1, (i_chan-1)*input_os_keep+1:(i_chan)*input_os_keep) = squeeze(freq_chan_pol_keep(1, 1, :));
      end
      % if max(stitched_freq(:)) > max_fft
      %   max_fft = max(stitched_freq(:));
      % end
      % FFTW does not do normalization with the inverse transform
      stitched_time = ifft(stitched_freq, output_fft_length) .* output_fft_length ;
      stitched_time_keep = stitched_time(1, output_discard.pos+1:end-output_discard.neg);
      idx = (i_part-1) * output_sample_step;
      inverted(1, i_pol, idx+1:idx+output_sample_step) = stitched_time_keep;

      % stitched_time = ifft(stitched_freq, output_fft_length); % .* output_fft_length ;
      % stitched_time_keep = stitched_time;
      % idx = (i_part-1) * output_fft_length;
      % inverted(1, i_pol, idx+1:idx+output_fft_length) = stitched_time_keep;
    end
  end
  % fprintf('max(data_in): %f\n', max(data_in(:)));
  % fprintf('max_fft: %f\n', max_fft);
  % fprintf('max(inverted): %f\n', max(inverted(:)));

end  % end function dspsr_inversion



function F1=fine_chan_proc(data_in, Nin, chan, OS_Nu, OS_De, equalise_ripple_, verbose_)

  % Reads one block of length Nin from a fine channel of the PFB, performs a
  % forward FFT and discards the oversampled portions (transition bands).
  % Optionally applies pass-band equalisation (de-ripple).
  % Optionally applies a phase shift of value that depends on the channel number.

  % savefile = fname_out;
  %
  % fid = fopen(fname_in);

  % Shift starting point for reading file (8 bytes per complex sample)
  % fseek(fid, input_offset*8, 'cof');

  % Read stream of complex voltages, forming a single column
  % Vstream = single(fread(fid, 2*Nin, 'single'));

  % if feof(fid)
  %     error('Error - hit end of input file!');
  % end;

  % Parse real and imag components
  % Vstream = reshape(Vstream, 2, []);
  % Vstream = complex(Vstream(1,:), Vstream(2,:));
  %
  % Vdat = reshape(Vstream, 1, []);

  equalise_ripple = 0;
  if exist('equalise_ripple_', 'var')
    equalise_ripple = equalise_ripple_;
  end

  verbose = 0;
  if exist('verbose_', 'var')
    verbose = verbose_;
  end

  if verbose
    fprintf('find_chan_proc: Nin: %d\n', Nin);
    fprintf('find_chan_proc: chan: %d\n', chan);
    fprintf('find_chan_proc: OS_Nu: %d\n', OS_Nu);
    fprintf('find_chan_proc: OS_De: %d\n', OS_De);
    fprintf('find_chan_proc: equalise_ripple: %d\n', equalise_ripple);
  end

  Vdat = reshape(data_in, 1, []);
  % Optional phase shift - 8 channels only at present
  phase_shift = 1.0;
  phase_shift_arr = [0,...
    1j,...
    0.5 + (sqrt(3.0)/2.0)*1j,...
    sqrt(3.0)/2.0 + 0.5i,...
    1,...
    sqrt(3.0)/2.0 - 0.5i,...
    0.5 - (sqrt(3.0)/2.0)*1j,...
    -1j
  ];

  % phase_shift = phase_shift_arr(chan);

  % Forward FFT
  F1 = fftshift(fft(Vdat(1,:).*phase_shift, Nin));
  F1 = reshape(F1, Nin, 1);
  % Keep only the pass-band
  discard = (1.0 - (OS_De/OS_Nu))/2.0;

  % discard low frequencies.
  % This is actually discarding high frequency components -- DCS
  F1 = F1(round(discard*Nin)+1:round((1.0-discard)*Nin));

  if verbose
    fprintf('fine_chan_proc: discard region: %d\n', discard);
    fprintf('fine_chan_proc: size(F1): [%d, %d]\n', size(F1));
  end

  % Optionally equalise PFB pass-band ripple
  if equalise_ripple
      % load PFB prototype filter transfer function
      load('config/TF_points.mat');
      figure;
      subplot(211); plot(abs(H0))
      size_H0 = size(H0);
      passbandLength = round(size_H0(1) / 8);
      sub_H0 = ones(passbandLength*2 + 1, 1);
      for ii = 1:passbandLength
          sub_H0(ii, 1) = 1.0/abs(H0(passbandLength-ii+2));
          sub_H0(passbandLength+ii, 1) = 1.0/abs(H0(ii+1));
      end;
      subplot(212); plot(sub_H0);
      pause;

      % size(H0)
      % Nin
      % use just the baseband passband section of transfer function
      % - apply to both halves of channel
      passband_len = (Nin/2)*OS_De/OS_Nu;
      for ii = 1:passband_len,
          F1(ii) = F1(ii)/abs(H0(passband_len-ii+2));
          F1(passband_len+ii) = F1(passband_len+ii)/abs(H0(ii+1));
      end;
  end;

  % save(savefile,'F1');
  %
  % fclose(fid);

end % end of fine_chan_proc

function z1=invert(fine_channel_data, Nin, n_chan, OS_Nu, OS_De, diagnostic_plots_, incremental_save_, verbose_)

  % Combines mutiple sub-channel pass-band chunks (from an oversampled PFB
  % that have had their transition bands discarded) into a single contiguous
  % block, then inverse FFTs.  The output is compared with the original in
  % both the time and frequency domains.
  % Ian Morrison
  % 21-4-16
  % Dean Shaff (updated)
  % 10-01-2019

  diagnostic_plots = 0;
  if exist('diagnostic_plots_', 'var')
    diagnostic_plots  = diagnostic_plots_;
  end

  incremental_save = 0;
  if exist('incremental_save_', 'var')
    incremental_save  = incremental_save_;
  end

  verbose = 0;
  if exist('verbose_', 'var')
    verbose  = verbose_;
  end


  % load and concatenate the chunks
  % chan = 1;
  % load(strcat(fname_in,int2str(chan),'.mat'));
  % FFFF = F1((length(F1)/2)+1:length(F1)); % upper half is first part of FFFF
  % for chan = 2 : n_chan,
  %     load(strcat(fname_in,int2str(chan),'.mat'));
  %     FFFF = [FFFF; F1];
  % end;
  % chan = 1;
  % load(strcat(fname_in,int2str(chan),'.mat'));
  % FFFF = [FFFF; F1(1:(length(F1)/2))]; % lower half is last part of FFFF
  %
  % len = length(FFFF);
  %
  % save('N_channels','FFFF');
  FFFF = fine_channel_data(1, (Nin/2)+1:end);
  for chan=2:n_chan
    FFFF = horzcat(FFFF, fine_channel_data(chan,:));
  end
  FFFF = horzcat(FFFF, fine_channel_data(1, 1:(Nin/2)));

  % FFFF_pos = fine_channel_data(1, (Nin/2)+1:end);
  % FFFF_neg = fine_channel_data(1, 1:(Nin/2));
  %
  % for chan=2:n_chan
  %   FFFF_pos = horzcat(FFFF_pos, fine_channel_data(chan,(Nin/2)+1:end));
  %   FFFF_neg = horzcat(fine_channel_data(chan, 1:(Nin/2)), FFFF_neg);
  % end
  %
  % FFFF = horzcat(FFFF_pos, FFFF_neg);

  % FFFF = [];
  % for chan=1:n_chan
  %   FFFF = horzcat(FFFF, fine_channel_data(chan,:));
  % end

  len = length(FFFF);
  fprintf('output_fft_length: %d', len)

  if incremental_save
    save('N_channels','FFFF');
  end

  if diagnostic_plots
    figure;
    subplot(211); plot((1:len), abs(FFFF)); box on; grid on; title('FFFF Mag');
    subplot(212); plot((1:len), angle(FFFF)); box on; grid on; title('FFFF Phase'); xlabel('time');
  end

  % back transform
  % z1 = (ifft((FFFF), len))./(OS_Nu/OS_De);  % re-scale by OS factor
  z1 = ifft(FFFF, len);
  if incremental_save
    save('inverse_PFB', 'z1');
  end

  if diagnostic_plots
    figure;  % set(gcf,'Visible', 'off');
    subplot(211); plot((1:len),real(z1(1:len))); box on; grid on; title('z1 Real');
    subplot(212); plot((1:len),imag(z1(1:len))); box on; grid on; title('z1 Imag'); xlabel('time');

    figure;  % set(gcf,'Visible', 'off');
    subplot(211); plot((1:len),10.0*log10(abs(real(z1(1:len)))+1e-12)); box on; grid on; title('z1 Real - Log scale');
    axis([1 len -100 10]);
    subplot(212); plot((1:len),10.0*log10(abs(imag(z1(1:len)))+1e-12)); box on; grid on; title('z1 Imag - Log scale'); xlabel('time');
    axis([1 len -100 10]);
  end
end % end of invert
