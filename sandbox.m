% Let's see if we can read in the output of the PFB Channelization step
% In addition, let's see if we can plot something useful

pfb_output_filename = 'os_channelized_pulsar.dump';

% Let's define some known characteristics about this dataset. 

hdrsize = 4096; %Header size
hdrtype = 'uint8'; % Data type for header ('uint8' = byte)
ntype = 'single'; % Data type for each element in a pair ('single' = float)
nseries = 80;

npol = 2; 
M = 7;
Nin = M*(2^14);
 

% There is a header in this file, but it is empty.

output_id = fopen(pfb_output_filename, 'r');

hdr = fread(output_id, hdrsize, hdrtype);

disp('hdr size:');
disp(size(hdr));

% Now let's get the actual data.

dat = fread(output_id, [2*npol*Nin/M, nseries], ntype);

disp('data size:');
disp(size(dat));

dat = reshape(dat, [2*npol, Nin/M, nseries]);

disp('new data size:');
disp(size(dat));

x = linspace(0, Nin/M, Nin/M); 

pol1 = complex(squeeze(dat(1,:,:)), squeeze(dat(2,:,:)));
pol2 = complex(squeeze(dat(3,:,:)), squeeze(dat(4,:,:)));

disp('pol1 size:')
disp(size(pol1))

subplot(2, 2, 1);
plot(x, abs(pol1(:, 1).^2))
subplot(2, 2, 2);
plot(x, angle(pol1(:, 1)))
subplot(2, 2, 3)
plot(x, real(pol1(:, 1)))
subplot(2, 2, 4) 
plot(x, imag(pol1(:, 1)))

% plot(x, reshape(dat(1,1,:), [1, Nin/M]));

fclose(output_id);