function PFBinversion (filename_in, verbose_)

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
end % end of PFBinversion
