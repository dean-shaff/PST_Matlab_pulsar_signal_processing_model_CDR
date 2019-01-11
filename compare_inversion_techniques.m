function compare_inversion_techniques(inverted_filename1, inverted_filename2)

  inverted_filenames = {inverted_filename1, inverted_filename2};
  dsets = {};
  n_pol = 2;
  for i=1:length(inverted_filenames)
    filename = inverted_filenames{i};
    if endsWith(filename, '.mat')
      % read in inverted data (saved in .mat matlab file)
      fprintf('Using .mat file\n');
      load(filename);
      data_inverted = inverted;

    elseif endsWith(filename, '.dump')
      % read in inverted dat from dspsr dump
      fprintf('Using .dump file\n');
      hdr_map = read_header(filename, containers.Map());
      hdr_size = str2num(hdr_map('HDR_SIZE'));
      n_dim_inv = str2num(hdr_map('NDIM'));
      n_pol_inv = str2num(hdr_map('NPOL'));
      n_bit_inv = str2num(hdr_map('NBIT'));

      fid_inv = fopen(filename);
      % skip header
      fread(fid_inv, hdr_size, 'uint8');

      data_inverted = fread(fid_inv, 'single');
      % figure;
      % subplot(1, 1, 1);
      %   plot((1:length(data_inverted)), data_inverted);
      %   pause

      data_inverted = reshape(data_inverted, 1, []);
      data_inverted = data_inverted ./ 229376;

      fclose(fid_inv);

      if n_dim_inv == 2
        fprintf('compare_inversion: inverted data is complex\n');
        data_inverted = reshape(data_inverted, n_dim_inv, []);
        data_inverted = complex(data_inverted(1, :), data_inverted(2, :));
      else
        fprintf('compare_inversion: inverted data is real\n');
        throw MException('Real data not supported');
      end
    end
    data_inverted = transpose(reshape(data_inverted, n_pol, []));
    dsets{i} = data_inverted;


  end
  len = length(dsets{1});
  len = round(0.2 * len);
  figure;
  set(gcf, 'Position', [10, 10, 1210, 810]);
  for pol=1:n_pol
    x = 1:len;
    incr = pol - 1;
    real_sim = real(dsets{1}(1:len, pol));
    imag_sim = imag(dsets{1}(1:len, pol));
    real_inv = real(dsets{2}(1:len, pol));
    imag_inv = imag(dsets{2}(1:len, pol));

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
  % Let's determine if the two arrays are _identical_.
  sqr_diff = sum(abs((dsets{1}(1:len,:) - dsets{2}(1:len,:)).^2));
  fprintf('square difference: %.9f\n', sqr_diff);


end  % end compare_inversion_techniques
