function [data, hdr_map]=load_simulated_pulsar_dump (filename)
  % read in input data
  hdr_map = read_header(filename, containers.Map());
  hdr_size = str2num(hdr_map('HDR_SIZE'));
  n_dim = str2num(hdr_map('NDIM'));
  n_pol = str2num(hdr_map('NPOL'));

  fid_sim = fopen(filename);
  % skip header
  fread(fid_sim, hdr_size, 'uint8');

  data = fread(fid_sim, 'single');
  data = reshape(data, 1, []);
  fclose(fid_sim);

  if n_dim == 2
    fprintf('compare_inversion: pulsar data is complex\n');
    % data = reshape(data, n_dim, []);
    % data = complex(data(1, :), data(2, :));
    data = reshape(data, 2*n_pol, []);
    size_data = size(data);
    dat_temp = zeros(n_pol, size_data(end));
    dat_temp(1, :) = complex(data(1,:), data(2,:));
    dat_temp(2, :) = complex(data(3,:), data(4,:));
    data = transpose(dat_temp);
  else
    fprintf('compare_inversion: pulsar data is real\n');
    throw MException('Real data not supported');
  end
end % end load_simulated_pulsar_dump
