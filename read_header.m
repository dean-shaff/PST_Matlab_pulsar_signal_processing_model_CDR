function hdr_map = read_header(hdr_filename_or_id, map)
  hdr_map = map;

  hdr_file_id = hdr_filename_or_id;
  if isstring(hdr_filename_or_id) || ischar(hdr_filename_or_id)
    if exist(hdr_filename_or_id, 'file')
      hdr_file_id = fopen(hdr_filename_or_id, 'r');
    else
      fprintf('%s does not exist\n', hdr_filename_or_id);
      return;
    end
  end

  formatter = '%c';
  headerStr = fscanf(hdr_file_id, formatter);
  fclose(hdr_file_id);

  headerLines = strsplit(headerStr, '\n');
  for i=1:length(headerLines)
    line = headerLines{i} ;
    if startsWith(line, '#')
      continue
    end

    keyValPair = strsplit(headerLines{i});
    if length(keyValPair) > 1
      hdr_map(keyValPair{1}) = keyValPair{2};
    end
  end
  utcnow = datetime('now', 'TimeZone', 'UTC');
  utcnow = datestr(utcnow, 'yyyy-mm-dd-HH:MM:ss');
  hdr_map('UTC_START') = utcnow;

  % get number of channels
  nchan = 1;
  if isKey(hdr_map, 'NCHAN')
    nchan = str2num(hdr_map('NCHAN'));
  end

  % get sampling rate
  tsamp = 0.0125;
  if isKey(hdr_map, 'TSAMP')
    tsamp = str2num(hdr_map('TSAMP'));
  end

  Nu_val = 1; % numerator value
  De_val = 1; % denominator value
  % Over-Sampling Factor
  if isKey(hdr_map,'OS_FACTOR')
    try
      splitInput = strsplit(hdr_map('OS_FACTOR'),'/');
      Nu_val = str2num(splitInput{1});
      De_val = str2num(splitInput{2});
    catch
      fprintf('Error: oversampling factor is not a fraction\nIgnore if OS_FACTOR is 1\n');
    end
  end
  fprintf('oversampling factor %i/%i\n', Nu_val, De_val);
  fprintf('sample rate before:\n\t%.10f\n', tsamp);
  tsamp = tsamp*nchan*(De_val/Nu_val);
  fprintf('updated sample rate:\n\t%.10f\n', tsamp);

  hdr_map('TSAMP') = num2str(tsamp);
end
