file_name = 'sanity_check.dump';

data = 1:0.1:10;

disp(size(data))

id = fopen(file_name, 'w');

fwrite(id, data, 'double');

fclose(id);
