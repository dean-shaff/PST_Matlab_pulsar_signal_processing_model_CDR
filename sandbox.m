% Let's see if we can read in the output of the PFB Channelization step
% In addition, let's see if we can plot something useful

pfb_output_filename = 'data/os_channelized_pulsar.dump.ref';
all_channels_output_filename = 'data/full_channelized_pulsar.dump';

% Let's define some known characteristics about this dataset. 

hdrsize = 4096; %Header size
hdrtype = 'uint8'; % Data type for header ('uint8' = byte)
ntype = 'single'; % Data type for each element in a pair ('single' = float)
nseries = 2;
f_sample_in = 80;

npol = 2; 
L = 8;
M = 7;
Nin = M*(2^14);
 

% There is a header in this file, but it is empty.

output_id = fopen(all_channels_output_filename, 'r');

hdr = fread(output_id, hdrsize, hdrtype);

disp('hdr size:');
disp(size(hdr));

% Now let's get the actual data.

dat_flat = fread(output_id, npol*L*(Nin/M)*2*nseries, ntype);
disp('data size:');
disp(size(dat));
fclose(output_id);
dat = reshape(dat_flat, [npol, L, Nin/M, 2, nseries]);
disp('data size:');
disp(size(dat));

x = linspace(0, 1, Nin/M); 

pol1 = complex(squeeze(dat(1,3,:,1, 1)), squeeze(dat(1,3,:,2, 1)));
% pol2 = complex(squeeze(dat(1,2,1,3,:)), squeeze(dat(1,2,2,3,:)));

disp('pol1 size:')
disp(size(pol1))

idx = 1;

nbin = Nin/M;

pol1_real = zeros(nbin,1);
pol1_imag = zeros(nbin,1);

i_channel = 3 ;
i_pol = 1;
i_series = 1;


for ii=1:nbin
   idx_real = (i_pol - 1) + L*
   idx_imag = 
%    incr = (i_series - 1)*(2*L*nbin*2) + (i_pol - 1)*(L*nbin*2) + (i_channel - 1)*(nbin*2);
   pol1_real(ii) = dat_flat(incr + (ii-1)*2 + 0);
   pol1_imag(ii) = dat_flat(incr + (ii-1)*2 + 1);
end

pol1_flat = complex(pol1_real, pol1_imag);
disp('pol1_flat size:');
disp(size(pol1_flat));

subplot(4, 1, 1);
plot((1:Nin/M), real(pol1(:, idx))); box on; grid on;
subplot(4, 1, 2);
plot((1:Nin/M), imag(pol1(:, idx))); box on; grid on;
subplot(4, 1, 3);
plot((1:Nin/M), real(pol1_flat)); box on; grid on;
subplot(4, 1, 4);
plot((1:Nin/M), imag(pol1_flat)); box on; grid on;

