function fname = signalgen(nseries, noise, signal_type_, verbose_)
% Generates a file containing dual noise vectors with phase-dependent
% partial polarization. File is 32-bit floating point with polarizations
% interleaved at each time step.

% DATA SETTINGS
%
% fname     - Ouput filename
% headerFile - A dada-style pulsar data header file
% hdrsize   - Header size
% hdrtype   - Data type for header ('uint8' = byte)
% ntype     - Data type for each element in a pair ('single' = float)
% Nout      - Length of each output vector
% nbins     - Number of bins within a pulse period
% npol      - Number of polarizations (should always be 2 when calc Stokes)
% nseries   - Number of forward FFT's to perform
% noise     - Set to 0.0 for no noise, 1.0 for noise (max(S/N)=1)
% dformat   - Specifies conversion TO real or complex data
% shift     - Select whether an fftshift is used before the inverse FFT
%             (don't shift if PFB is in the signal chain)
%
% INSTRUMENT SETTINGS
% f0        - Centre frequency (MHz)
% f_sample_out - Sampling frequency of output data (MHz)
%
% PULSAR SETTINGS
% Dconst    - Dispersion constant, s.MHz^2/(pc/cm^3)
% DM        - Dispersion measure, pc/cm^3
% pcal      - Pulsar period (s) and other params in a structure
% t0        - Absolute time (in seconds) of first time element
%
% OUTPUTS:
% --------
%
%    fname -  file containing two interleaved floating point test vectors
%
% Description:
% ------------
% Generates a file containing dual noise vectors with phase-dependent
% partial polarization. File is 32-bit floating point with polarizations
% interleaved at each time step.
%
% Changes:
% --------
%
% Author           Date         Comments
% ---------------  -----------  ----------------------------------------
% D. Hicks         04-Jul-2014  Original version
% I. Morrison      31-Jul-2015  Added noise parameter
%                               Added optional fftshift before inverse FFT
% R. Willcox       07-Sep-2018  Added header read-in
% ----------------------------------------------------------------------

%=============
signal_type = 'pulsar';
if exist('signal_type_', 'var')
  signal_type = signal_type_
end


verbose = 1;
if exist('verbose_', 'var')
  verbose = verbose_
end


%=============
% Default settings for variables that might be found in a header file
headerFile = 'config/gen.header'; % Use a better name

hdrsize = 4096; % Header size
npol = 2; % Number of polarizations (should always be 2 when calc Stokes)
f0 = 1405; % Centre frequency (MHz)
T_pulsar = .00575745; % Pulsar period

% Set bandwidth - default is 8 x 10 MHz, for testing with 8-channel channelizer
f_sample_out = 80; % Sampling frequency of output (MHz)

% Multiplying factor going from input to output type
dformat = 'complextoreal'; %specifies conversion TO real or complex data
% dformat = 'complextocomplex'; %specifies conversion TO real or complex data

%=============
% Get data from header
hdr_map = read_header(headerFile, containers.Map());
% Number of polarizations
if isKey(hdr_map, 'NPOL') npol = str2num(hdr_map('NPOL')); end
% Centre frequency (MHz)
if isKey(hdr_map, 'FREQ') f0 = str2num(hdr_map('FREQ')); end
% Pulsar period
if isKey(hdr_map, 'CALFREQ') T_pulsar = 1.0/str2num(hdr_map('CALFREQ')); end

if isKey(hdr_map, 'BW') f_sample_out = str2num(hdr_map('BW')); end

if isKey(hdr_map, 'NDIM')
  nchan = hdr_map('NDIM');
  switch nchan
    case '1'
      dformat = 'complextoreal';
    case '2'
      dformat = 'complextocomplex';
  end
end

utcnow = datetime('now', 'TimeZone', 'UTC');
utcnow = datestr(utcnow, 'yyyy-mm-dd-HH:MM:ss');
hdr_map('UTC_START') = utcnow;

% Set bandwidth
% if isKey(hdr_map,'BW') f_sample_out = (-1)*str2num(hdr_map('BW')); end % Sampling frequency of output (MHz)

%=============
% Data which is not specified in the header

% Data settings
hdrtype = 'uint8'; % Data type for header ('uint8' = byte)
ntype = 'single'; % Data type for each element in a pair ('single' = float)
Nout = 2^20; %Length of each output vector
nbins = 2^10; % Number of bins within a pulse period
shift = 0; % performs an fftshift before the inverse FFT

switch dformat
    case 'complextoreal'
        Nmul = 2;
        NperPol = 1;
    case 'complextocomplex'
        Nmul = 1;
        NperPol = 2;
    otherwise
        warning('Conversion should be complextoreal or complextocomplex.')
end

% Pulsar settings
Dconst = 4.148804E3; % s.MHz^2/(pc/cm^3)
DM = 2.64476; % pc/cm^3
%DM = 2.64476*40; % pc/cm^3
pcal = struct('a',T_pulsar,'b',0.0);% Pulsar period (s) and other params
t0 = 0.0; % Absolute time (in seconds) of first time element

Tout = 1/abs(f_sample_out)*1E-6; % Sample spacing, or interval of output (seconds)
Tout = Tout / Nmul;
% df = f_sample_out/Nmul; % Bandwidth/Nyquist frequency (MHz)
df = f_sample_out;
% Tin = Tout*Nmul; % Time spacing between input data elements
Tin = Tout;
Nin = Nout/Nmul; % Number of data elements in input time series
Pmul = 1/Nmul; % Power multiplication factor for all but the DC channel




%===============
% Create the dispersion kernel and determine the number of elements to be
% clipped off the beginning and end.
frange = [-df/2, df/2] + f0;

% Get matrix to perform dispersion on complex array
[H, ~, n_hi, n_lo] = ...
         dispnmatrix(frange, abs(df), Nin, 1, Dconst*DM, Tin, sign(df));

% Calculate the number of elements in the clipped input array
nclip_in = Nin - n_lo - n_hi;
% Calculate number of elements in the clipped output array
nclip_out = Nout - n_lo*Nmul - n_hi*Nmul;

frac_lost = (n_lo + n_hi)/Nin; % fraction of array that's lost
fprintf('Lost fraction of time series = %f\n', frac_lost);
fprintf('Time series length = %f s\n', nclip_in*Tin);

% ================
% if using impulse option, set the width and offset
impulse_width = 1;
impulse_offset = round(nclip_out / 4);
%===============
% print out diagnostic message
if verbose
  fprintf('Dconst: %f\n',Dconst);
  fprintf('DM: %f\n',DM);
  fprintf('f_sample_out: %.1f\n', f_sample_out);
  fprintf('T_pulsar: %.10f\n', T_pulsar);
  fprintf('Tout: %.10f\n',Tout);
  fprintf('df: %.10f\n',df);
  fprintf('Nin: %d\n',Nin);
end
% return
%===============
% Calculate phase-dependent Stokes parameters and coherency matrix
% using the rotating vector model
[~, J] = rotvecmod(nbins,noise);

% Vector of relative times
trel = (0:Nin-1)*Tin;

%===============
% current_branch = git_current_branch();
if strcmp(signal_type, 'pulsar')
  fname = sprintf('data/simulated_pulsar.noise_%.1f.nseries_%d.ndim_%d.dump', noise, nseries, NperPol);
elseif strcmp(signal_type, 'impulse')
  fname = sprintf('data/impulse.noise_%.1f.nseries_%d.ndim_%d.dump', noise, nseries, NperPol)
end
% Open file for writing
hdr_map('TSAMP') = num2str(Tout * 1e6); % this is sampling interval, in microseconds
hdr_map('NDIM') = num2str(NperPol);

fid = fopen(fname, 'w');
write_header(fid, hdr_map);
fid = fopen(fname, 'a');
%=============
% Random vectors
if strcmp(signal_type, 'impulse')
  N = nclip_out*nseries;
  fprintf('N=%d\n', N)
  % t = 1:nclip_out*nseries;
  % t = linspace(0,1e5,nclip_out*nseries);
  % z = complex(zeros(nclip_out*nseries, npol, 'single'));
  % z(:, 1) = sin(t); %  + 1j*sin(t);
  % z(:, 2) = sin(t); %  + 1j*sin(t);
  freqVector = complex(zeros(N,npol,'single'));
  freqVector(round(N/20),:) = 1.0;


  z = complex(zeros(N, npol, 'single'));
  z(:, 1) = ifft(freqVector(:, 1)).*N;
  z(:, 2) = ifft(freqVector(:, 2)).*N;

  % figure;
  % subplot(211); plot((1:N),real(z(1:N,1))); box on; grid on; title('sig Mag');
  % subplot(212); plot((1:N),imag(z(1:N,1))); box on; grid on; title('sig Phase'); xlabel('frequency');

  z = [real(z(:,1)), imag(z(:,1)), real(z(:,2)), imag(z(:,2))];

  dat = reshape(transpose(z),2*npol*nclip_out*nseries,1);
  fwrite(fid, dat, ntype);
  fclose(fid);
  return
  % Introduce an impulse in z
  % z1(impulse_offset:impulse_offset+impulse_width-1,1) = 1 + 1i;
  % z1(impulse_offset:impulse_offset+impulse_width-1,2) = 1 + 1i;
  %
  % z1clip = ifft(z1(:, 1)).*nclip_out;
  % z2clip = ifft(z1(:, 2)).*nclip_out;
  % freqVector = complex(zeros(Nout,1,'double'));
  % freqVector(3) = 1.0;
  %
  % figure;
  % subplot(211); plot((1:Nout),abs(freqVector(1:Nout,1))); box on; grid on; title('sig Mag');
  % subplot(212); plot((1:Nout),phase(freqVector(1:Nout,1))); box on; grid on; title('sig Phase'); xlabel('frequency');
  %
  % z1 = ifft(freqVector).*Nout;

end


prev_bytes = 1;
for ii = 1:nseries,
    % for b=1:prev_bytes
    %   fprintf('\b');
    % end
    prev_bytes = fprintf('\nLoop # %i of %i\n', ii, nseries);
    % Time vector
    if ii == 1,
        tt = t0 - n_hi*Tin + trel;
    else
        tt = ttclip(end) - (n_hi-1)*Tin + trel;
    end;

    tindex = findphase(tt, nbins, pcal);
    index = unique(tindex);

    % Initialize data vector for this series
    z = zeros(Nin, npol, 'single');
    %iL = 1; %Starting index when looping through phases

    % Loop through groups of data that share the same phase. Random data
    % in each group are generated from the same coherency matrix

    for jj = 1:length(index),
        %Get coherency matrix for this pulsar phase
        Jcoh = [J(index(jj),1), J(index(jj),3); ...
                J(index(jj),2), J(index(jj),4)];

        % Indices of elements with a given phase
        iphase = find(tindex == index(jj));
        nL = length(iphase);

        %Generate two randomly-phased, unit-length phasors
        %z0 = exp(complex(0,1)*2*pi()*rand(nL,npol));
        z0 = sqrt(0.5)*[complex(randn(nL,1),randn(nL,1)), ...
                        complex(randn(nL,1),randn(nL,1))];

        %Generate covariant vectors via Cholesky decomposition
        zjj = z0*chol(Jcoh, 'upper');
        %z = transpose(chol(Jcoh, 'lower')*transpose(z0)); %alternative

        % Concatenate with data from other phases
        z(iphase, :) = zjj;
        %iL = iL + nL; % increment to next starting index in z
    end;

    % Forward FFT
    f1a = fft(z(:,1), Nin);
    f2a = fft(z(:,2), Nin);

    % Element-wise multiplication by dispersion matrix.
    f1a = f1a .* H;
    f2a = f2a .* H;

    % If complextoreal, then create a Hermitian array
    switch dformat
        case 'complextoreal'
            %Create Hermitian vector
            f1 = [real(f1a(1)); f1a(2:Nin)*Pmul; ...
                  imag(f1a(1)); flipud(conj(f1a(2:Nin)))*Pmul];
            f2 = [real(f2a(1)); f2a(2:Nin)*Pmul; ...
                  imag(f2a(1)); flipud(conj(f2a(2:Nin)))*Pmul];
        otherwise
            f1 = f1a;
            f2 = f2a;
    end;

    % Inverse FFT
    % Optionally include an fftshift before the inverse FFT, as needed
    if shift == 1,
        f1 = fftshift(f1);
        f2 = fftshift(f2);
    end;
    z1 = ifft(f1, Nout);
    z2 = ifft(f2, Nout);

    % Remove convolution overlap region
    ttclip = tt(1+n_hi : Nin-n_lo);
    z1clip = z1(1+n_hi*Nmul : Nout-n_lo*Nmul);
    z2clip = z2(1+n_hi*Nmul : Nout-n_lo*Nmul);

    % Interleave polarizations into a single vector
    switch dformat
        case 'complextoreal'
            z = [z1clip, z2clip];
            dat = reshape(transpose(z),npol*nclip_out,1);
        case 'complextocomplex'
            % z = [real(z1clip), real(z2clip), imag(z1clip), imag(z2clip)];
            z = [real(z1clip), imag(z1clip), real(z2clip), imag(z2clip)];
            dat = reshape(transpose(z),2*npol*nclip_out,1);
    end

    %Write vector to file
    fwrite(fid, dat, ntype);
end

fclose(fid);
return

end



function [S, J, p] = rotvecmod(N, noise, showplot)
% Rotating vector model for pulsar emission

if ~exist('N','var'),
    N = 1024;
end;

esig = 5. ; % emission half angle (polar angle, degrees)
epeak = 0. ; % emission peak angle (polar angle, degrees)
flin = 0.3; % linear polarized fraction amplitude
%flin = 1; % linear polarized fraction amplitude % Reinhold test

zeta = 30.; % observing angle (degrees) relative to rotation axis
alpha = 40.; % magnetic axis (degrees) relative to rotation axis

pmin = -180.;
pmax = 180.;

% Angle of rotation: p=0 for aligned dipole.
% This is equivalent to pulsar longitude or phase
p = transpose(linspace(pmin, pmax, N));

% Polarization angle w.r.t projected rotation axis from observing direction
%psi = atand(sind(alpha)*sind(p)./(sind(zeta)*cosd(alpha) - ...
%    sind(alpha)*cosd(zeta)*cosd(p)));
psi = atan2d(sind(alpha)*sind(p),  ...
    (sind(zeta)*cosd(alpha) - sind(alpha)*cosd(zeta)*cosd(p)));

% Polar observation angle in magnetic axis reference frame
cosO = cosd(p)*sind(zeta)*sind(alpha) + cosd(alpha)*cosd(zeta);
tanO = sqrt(1./(cosO.^2)-1);

% Polar emission angle in magnetic axis reference frame
thetaE = atand(1.5*(sqrt(1+(8/9)*tanO.^2) - 1)./tanO);
%thetaE = atand(1.5*(-sqrt(1+(8/9)*tanO.^2) - 1)./tanO);

% Intensity (model-based assumption)
S0 = (1./sqrt(2*pi()*esig^2))*exp(-(thetaE-epeak).^2/(2.*esig^2));
S0 = S0/max(S0); %normalize max to 1

% Linear polarization fraction (model-based assumption)
L = flin*S0.*cosd(thetaE);

% Other Stokes parameters
S1 = L.*cosd(2*psi);
S2 = L.*sind(2*psi);
S3 = -(1-flin)*S1; % Fake circular polarization to avoid zero signal
%S3 = single(zeros(N,1)); % Zero circular polarization component

% Add noise, typically such that max(S/N) = 1
S0 = S0 + noise;

% Normalize Stokes 4-vector so that S0 = 1.
factor = max(S0);
S0 = S0/factor;
S1 = S1/factor;
S2 = S2/factor;
S3 = S3/factor;

% Create Coherency matrix
Jxx = 0.5*(S0 + S1);
Jyy = 0.5*(S0 - S1);
Jxy = 0.5*(S2 + 1i*S3);
Jyx = 0.5*(S2 - 1i*S3);

% Plot results, if requested. Useful for debugging.
if exist('showplot','var'),
    clf();

    subplot(2,2,1);
    plot(p, transpose([S0, S1, S2, S3]));
    legend('S0', 'S1', 'S2', 'S3');
    xlabel('Longitude (degrees)','FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude','FontSize', 12, 'FontWeight', 'bold');
    set(gca,'FontSize', 12, 'FontWeight', 'bold');

    subplot(2,2,3);
    plot(S0, transpose([S1, S2]));
    hleg1 = legend('S1', 'S2');
    set(hleg1,'Location','NorthWest')
    axis([0, 2, -Inf, Inf]);
    xlabel('S0','FontSize', 12, 'FontWeight', 'bold');
    ylabel('S1 or S2','FontSize', 12, 'FontWeight', 'bold');
    set(gca,'FontSize', 12, 'FontWeight', 'bold');

    subplot(2,2,2);
    plot(p, transpose([Jxx, Jyy, real(Jxy), imag(Jxy)]));
    legend('Jxx', 'Jyy', 'Real(Jxy)', 'Imag(Jxy)');
    xlabel('Longitude (degrees)','FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude','FontSize', 12, 'FontWeight', 'bold');
    set(gca,'FontSize', 12, 'FontWeight', 'bold');

    subplot(2,2,4);
    plot(S1, S2, 'b');
    xlabel('S1','FontSize', 12, 'FontWeight', 'bold');
    ylabel('S2','FontSize', 12, 'FontWeight', 'bold');
    set(gca,'FontSize', 12, 'FontWeight', 'bold');
end;

S = [S0, S1, S2, S3];
J = [Jxx, Jyx, Jxy, Jyy];
return
end


% Function to pull observation parameters from a header file
function hdr_map = headerReadIn(headerFile, hdr_map)
% This reads a header file typical of pulsar observations
% (dada, cspsr2, etc.) and parses the parameters found there.
% When the .dump file is prepared, the header and .dump
% files should be catted together with a buffer of nulls to
% fill up the the header, before being read into DSPSR

if exist(headerFile, 'file')
    'The header file was found';
    fheaderFile = fopen(headerFile, 'r');
    formatSpec = '%c'; %collects all chars
    headerString = fscanf(fheaderFile, formatSpec);
    headerLines = strsplit(headerString, '\n'); % output is a row vector where each element is a line from the header file

    for i=1:length(headerLines)
       tempMap = strsplit(headerLines{i}); % Parse lines along whitespace
       if length(tempMap) > 1
          hdr_map(tempMap{1}) = tempMap{2};
       end
    end
    fclose(fheaderFile);
else
    'The header file was not found'
end
return
end
