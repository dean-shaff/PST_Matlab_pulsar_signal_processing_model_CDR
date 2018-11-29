function write_header(out_filename_or_id, hdr_map)
    hdr_str = "";

    for k=keys(hdr_map)
      key_val = sprintf('%s %s', k{1}, hdr_map(k{1}));
      hdr_str = strcat(hdr_str, key_val);
      if is_octave()
        hdr_str = strcat(hdr_str, char(10));
      else
        hdr_str = hdr_str + newline;
      end
    end
    hdr_size = str2num(hdr_map('HDR_SIZE'));
    hdr_char = char(hdr_str);
    n_remaining = hdr_size - length(hdr_char);
    hdr_remaining = char('0' * zeros(n_remaining, 1));

    fid = out_filename_or_id;
    if isstring(out_filename_or_id) || ischar(out_filename_or_id)
      fid = fopen(out_filename_or_id, 'w');
    end
    fwrite(fid, hdr_char, 'char');
    fwrite(fid, hdr_remaining, 'char');
    fclose(fid);
end
