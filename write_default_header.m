function write_default_header (fid, hdrsize, dformat)
    % We're looking to write a header with the following content:
    % HDR_VERSION 1.0
    % HDR_SIZE 4096
    % TELESCOPE Parkes
    % SOURCE  J0437-4715
    % FREQ 1405
    % BW 40 or 80
    % TSAMP 0.0125
    % UTC_START yyyy-mm-dd-hh:mm:ss
    % OBS_OFFSET 0
    % NPOL 2
    % NDIM 1 or 2
    % NBIT 32
    % CALFREQ 11.123
    % MODE    PSR
    % INSTRUMENT   dspsr

    switch dformat
      case 'complextoreal'
          ndim = '1';
          BW='40';
      case 'complextocomplex'
          ndim = '2';
          BW='80';
    end

    % for the following to work on ozstar you have to set the TZ environment
    % variable to whatever the current time zone is.

    utcnow = datetime('now', 'TimeZone', 'UTC');
    utcnow = datestr(utcnow, 'yyyy-mm-dd-HH:MM:ss');

    hdr_map = containers.Map('KeyType', 'char', 'ValueType', 'char');

    hdr_map('HDR_VERSION') = '1.0';
    hdr_map('HDR_SIZE') = num2str(hdrsize);
    hdr_map('TELESCOPE') = 'PARKES';
    hdr_map('SOURCE') = 'J0437-4715';
    hdr_map('FREQ') = '1405';
    hdr_map('BW') = BW;
    hdr_map('TSAMP') = '0.0125';
    hdr_map('UTC_START') = utcnow;
    hdr_map('OBS_OFFSET') = '0';
    hdr_map('NPOL') = '2';
    hdr_map('NDIM') = ndim;
    hdr_map('NBIT') = num2str(4*8);
    hdr_map('MODE') = 'PSR';
    hdr_map('INSTRUMENT') = 'dspsr';


    hdr_str = "";

    for k=keys(hdr_map)
      key_val = sprintf("%s %s", k{1}, hdr_map(k{1}));
      hdr_str = strcat(hdr_str, key_val);
      hdr_str = hdr_str + newline;
    end

    % disp(hdr_str);

    hdr_char = char(hdr_str);
    n_remaining = hdrsize - numel(hdr_char);
    hdr_remaining = char('0' * zeros(n_remaining, 1));

    fwrite(fid, hdr_char, 'char');
    fwrite(fid, hdr_remaining, 'char');

end
