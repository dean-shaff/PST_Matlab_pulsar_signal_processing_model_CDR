% test the write_default_header.m script 

test_file_name = "test_header.dump";

fid = fopen(test_file_name, "w");

write_default_header(fid, 4096);

fclose(fid);