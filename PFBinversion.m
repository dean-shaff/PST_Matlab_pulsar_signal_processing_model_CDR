function PFBinversion (filename_in, verbose_)

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

  data_in = fread(fid_in,'single');
  fclose(fid_in);

  data_in = reshape(data_in, n_pol*n_dim, n_chan, []);
  if n_dim == 2
    data_in = reshape(data_in, n_dim, n_pol, n_chan, []);
    data_in = complex(data_in(1,:,:,:), data_in(2,:,:,:));
  else
    throw(MException('No support for real input data'));
  end

  size_data_in = size(data_in);
  % fprintf('size_data_in: %a\n', size_data_in)
  n_samples = size_data_in(end);

  os_keep_region = (OS_De/OS_Nu)*n_samples;
  input_offset = 0;
  equalise_ripple = 0;
  fine_channels = zeros(n_pol, n_chan, os_keep_region);

  % process fine channels
  for chan=1:n_chan
    for pol=1:n_pol
      fprintf('Processing fine channel %d, polarization %d\n', chan, pol);
       fine_chan_proc_res = fine_chan_proc(...
        data_in(1, pol, chan, :), n_samples, chan, OS_Nu, OS_De, input_offset, equalise_ripple, 1);
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

  % save inverted data
  inverted_file_name = strcat(filename_in, '.inverted.mat');
  save(inverted_file_name, 'inverted')

end % end of PFBinversion

function F1=fine_chan_proc(data_in, Nin, chan, OS_Nu, OS_De, input_offset_, equalise_ripple_, verbose_)

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
  input_offset = 0;
  if exist('input_offset_', 'var')
    input_offset = input_offset_;
  end

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
    fprintf('find_chan_proc: input_offset: %d\n', input_offset);
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

  if (0)
    phase_shift = phase_shift_arr(chan)
  end
  % Forward FFT
  F1 = fftshift(fft(Vdat(1,:).*phase_shift, Nin));
  F1 = reshape(F1, Nin, 1);
  % Keep only the pass-band
  discard = (1.0 - (OS_De/OS_Nu))/2.0;


  F1 = F1(round(discard*Nin)+1:round((1.0-discard)*Nin));

  if verbose
    fprintf('fine_chan_proc: discard region: %d\n', discard);
    fprintf('fine_chan_proc: size(F1): [%d, %d]\n', size(F1));
  end

  % Optionally equalise PFB pass-band ripple
  if(equalise_ripple)
      % load PFB prototype filter transfer function
      load('config/TF_points.mat');

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

  len = length(FFFF);

  if incremental_save
    save('N_channels','FFFF');
  end

  if diagnostic_plots
    figure;
    subplot(211); plot((1:len),abs(FFFF)); box on; grid on; title('FFFF Mag');
    subplot(212); plot((1:len),angle(FFFF)); box on; grid on; title('FFFF Phase'); xlabel('time');
  end

  % back transform
  z1 = (ifft((FFFF), len))./(OS_Nu/OS_De);  % re-scale by OS factor

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
