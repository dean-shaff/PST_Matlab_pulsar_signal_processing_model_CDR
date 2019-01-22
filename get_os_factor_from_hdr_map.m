function [OS_Nu, OS_De] = get_os_factor_from_hdr_map (hdr_map)
  OS_factor = hdr_map('OS_FACTOR');
  OS_factor_split = strsplit(OS_factor, '/');
  OS_Nu = str2num(OS_factor_split{1});
  OS_De = str2num(OS_factor_split{2});
end % end function get_os_factor_from_hdr_map
