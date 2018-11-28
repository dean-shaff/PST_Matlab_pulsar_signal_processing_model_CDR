function headerMap = read_header(headerFileName, map)
  headerMap = map
  if exist(headerFileName, 'file')
    headerFile = fopen(headerFileName, 'r');
    formatter = '%c';
    headerStr = fscanf(headerFile, formatter);
    headerLines = strsplit(headerStr, '\n');
    for i=1:length(headerLines)
      line = headerLines{i} ;
      if startsWith(line, '#')
        continue
      end

      keyValPair = strsplit(headerLines{i});
      if length(keyValPair) > 1
        headerMap(keyValPair{1}) = keyValPair{2};
      end
    end
    utcnow = datetime('now', 'TimeZone', 'UTC');
    utcnow = datestr(utcnow, 'yyyy-mm-dd-HH:MM:ss');
    headerMap('UTC_START') = utcnow;

    % get number of channels
    nchan = 1;
    if isKey(headerMap, 'NCHAN')
      nchan = str2num(headerMap('NCHAN'));
    end

    % get sampling rate
    tsamp = 0.0125;
    if isKey(headerMap, 'TSAMP')
      tsamp = str2num(headerMap('TSAMP'));
    end

    Nu_val = 1; % numerator value
    De_val = 1; % denominator value
    % Over-Sampling Factor
    if isKey(headerMap,'OS_FACTOR')
        splitInput = strsplit(headerMap('OS_FACTOR'),'/');
        Nu_val = str2num(splitInput{1});
        De_val = str2num(splitInput{2});
    end

    tsamp = tsamp*nchan*De_val/Nu_val;

    headerMap('TSAMP') = num2str(tsamp);

  end
  return
end
