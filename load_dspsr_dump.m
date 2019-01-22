function data=load_dspsr_dump (filename, OS_Nu_, OS_De_)

  OS_Nu = 1;
  if exist('OS_Nu_', 'var')
    OS_Nu = OS_Nu_;
  end

  OS_De = 1;
  if exist('OS_De_', 'var')
    OS_De = OS_De_;
  end

  % read in inverted dat from dspsr dump
  fprintf('Using .dump file\n');
  hdr_map = read_header(filename, containers.Map());
  hdr_size = str2num(hdr_map('HDR_SIZE'));
  n_dim = str2num(hdr_map('NDIM'));
  n_pol = str2num(hdr_map('NPOL'));
  n_bit = str2num(hdr_map('NBIT'));
  % OS_factor = hdr_map('OS_FACTOR');
  % OS_factor_split = strsplit(OS_factor, '/');
  % OS_Nu = str2num(OS_factor_split{1});
  % OS_De = str2num(OS_factor_split{2});

  fid = fopen(filename);
  % skip header
  fread(fid, hdr_size, 'uint8');

  data = fread(fid, 'single');
  % figure;
  % subplot(1, 1, 1);
  %   plot((1:length(data)), data);
  %   pause

  data = reshape(data, 1, []);

  % data = data ./ 229376;
  % if (OS_Nu / OS_De == 1)
  %   data = data ./ 262144;
  % else
  %   data = data ./ 229376;
  % end

  fclose(fid);

  if n_dim == 2
    fprintf('compare_inversion: inverted data is complex\n');
    data = reshape(data, n_pol*n_dim, []);
    size_data = size(data);
    data_samples = size_data(end);
    dat_temp = zeros(n_pol, data_samples);
    dat_temp(1, :) = squeeze(complex(data(1,:), data(2,:)));
    dat_temp(2, :) = squeeze(complex(data(3,:), data(4,:)));
    data = transpose(dat_temp);
    % data = reshape(data, n_dim, n_pol, []);
    % data = squeeze(complex(data(:, 1, :), data(:, 2, :)));
    % data = transpose(data);
  else
    fprintf('load_dspsr_dump: data is real\n');
    throw MException('Real data not supported');
  end

end % end load_dspsr_dump
