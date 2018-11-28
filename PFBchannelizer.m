function PFBchannelizer()
%
% Takes as input a data file generated by "signalgen.m", passes it through
% a polyphase filterbank (PFB) channelizer, then stores the output of one
% channel to file.  The type of PFB is selectable: either critically
% sampled or oversampled. The number of PFB channels is selectable.
% The PFB prototype filter is designed separately and its coefficients
% provided in a file.
%
% Inputs:
% -------
%
% fname_in  - Input filename
% headerFile - A dada-style pulsar signal header file
% fname_pfb - PFB prototype filter coefficients filename
%
% SETTINGS
%
% hdrsize   - Header size
% hdrtype   - Data type for header ('uint8' = byte)
% ntype     - Data type for each element in a pair ('single' = float)
% npol      - Number of polarizations (should always be 2 when calc Stokes)
% dformat   - Specifies conversion TO real or complex data
% Nin       - Length of input blocks to be processed
% f_sample_in - Sampling frequency of input data (MHz)
% nseries   - Number input blocks to process
% pfb_type  - Type of PFB: 0 for critically samples, 1 for oversampled
% L         - Number of PFB channels
% Nu        - Numerator of oversampling factor
% De        - Denominator of oversampling factor
% M         - PFB commutator length
% L_M       - PFB overlap length
% chan_no   - selected output channel number to store
%
% OUTPUTS:
% --------
%
% fname_out - Output filename
%
% Changes:
% --------
%
% Author           Date         Comments
% ---------------  -----------  ----------------------------------------
% I. Morrison      31-Jul-2015  Original version
% R. Willcox       07-Sep-2018  Added Over-Sampling factor
%                               Added Header read and write
%
% ----------------------------------------------------------------------

close all; clear all; clc;

% Input file name
fname_in = 'data/simulated_pulsar.dump';

% Output file name
%fname_out = 'cs_channelized_pulsar.dump';
fname_out = 'data/os_channelized_pulsar.dump';

% Clear out the output file
    % Various pieces append at different points,
    % so this ensures that the file starts clean every time
fid_out = fopen(fname_out, 'w');
fclose(fid_out);

% Header name
headerFile = 'config/gen.header';


%=======================================
% PFB parameters

% Define globals common also to CS_PFB() / OS_PFB() sub-functions
global L; global Nu; global M; global L_M; global fname_pfb;

%=============
% Default settings for variables that might be found in a header file

hdrsize = 4096; % Header size
npol = 2; % Number of polarizations (should always be 2 when calc Stokes)

% Set bandwidth - default is 8 x 10 MHz, for testing with 8-channel channelizer
f_sample_in = 80; % Sampling frequency of input (MHz)

% Multiplying factor going from input to output type
dformat = 'complextoreal'; %specifies conversion TO real or complex data
%dformat = 'complextocomplex'; %specifies conversion TO real or complex data

% PFB type
pfb_type = 1; % 0 for critically sampled, 1 for oversampled
% OverSampling
Nu = 8; %Numerator
De = 7; %Denominator


%=============
% Header settings for variables, where they exist
%
% Get data from header
headerMap = containers.Map; %empty map
headerMap = headerReadWrite(headerFile, fname_out, headerMap);

% Header size
if isKey(headerMap,'HDR_SIZE') hdrsize = str2num(headerMap('HDR_SIZE')); end
% Number of polarizations
if isKey(headerMap,'NPOL') npol = str2num(headerMap('NPOL')); end

% Set bandwidth
if isKey(headerMap,'BW') f_sample_in = (-1)*str2num(headerMap('BW')); end % Samplin    g frequency of input (MHz)

% Multiplying factor going from output to input type
%
% NDIM is 1 for real input data and 2 for complex input data
if isKey(headerMap,'NDIM')
    if str2num(headerMap('NDIM'))==1 dformat='realtocomplex';
    elseif str2num(headerMap('NDIM'))==2 dformat='complextocomplex';
    else warning('NDIM in header file should be 1 or 2.')
    end
end

% Over-Sampling Factor
if isKey(headerMap,'OS_FACTOR')
    splitInput = strsplit(headerMap('OS_FACTOR'),'/'); %what if OS_FACTOR is just 1?
    Nu = str2num(splitInput{1});
    De = str2num(splitInput{2});
    if (Nu == 1 && De == 1)
        pfb_type=0; %for CSing
    else
        pfb_type=1; %for OSing
    end
end

%=======================================
% Other parameters
hdrtype = 'uint8'; % Data type for header ('uint8' = byte)
ntype = 'single'; % Data type for each element in a pair ('single' = float)
chan_no = 3; % Particular PFB output channel number to store to file
nseries = 80; % Number of input blocks to read and process

%=============
% Initialisations

% PFB prototype OS factor and filter coefficients file name
if pfb_type == 0, % Critically sampled
    Os = 1;
    %fname_pfb = 'CS_Prototype_FIR_8.mat';
    fname_pfb = 'config/OS_Prototype_FIR_8.mat'; % Where is CS_Prototype_FIR.mat?
else % Over sampled
    Os = Nu/De; % Oversampling factor
    fname_pfb = 'config/OS_Prototype_FIR_8.mat';
end;

% Number of channels in filter-bank
L= 16;
M = L/Os; % Commutator Length
L_M = L-M; % Overlap
Nin = M*(2^14);  % Number of elements per input file read


% Set up parameters depending on whether incoming data is real or complex
switch dformat
    case 'realtocomplex'
        Nmul = 2; % Multiplying factor in converting from real to complex
        NperNin = 1; % 1 data point per polariz per incoming time step
    case 'complextocomplex'
        Nmul = 1;
        NperNin = 2; % 1 real + 1 imag point per pol per incoming time step
    otherwise
        warning('Conversion should be realtocomplex or complextocomplex.');
end


%==============
% Fill up the header completely by appending null characters, \0

% Get current size of output file (just the header so far)
hdrfile=dir(fname_out);
hdr_currentsize=hdrfile.bytes;

% Append the remainder as nulls
fid_out = fopen(fname_out, 'a');
for i=1:(hdrsize-hdr_currentsize)
    fprintf(fid_out,'\0');
end
fclose(fid_out);

%==============
% Prepare for main loop

% Open input file
fid_in = fopen(fname_in);

% Open output file
fid_out = fopen(fname_out, 'a');

% Initialise output
y2 = zeros(npol,L,Nin/M);


%===============
% Main loop
% Read input blocks and filter

for ii = 1 : nseries

    % Print loop number
    fprintf('Loop # %i of %i\n', ii, nseries);

    % Read stream of voltages into a single column
    Vstream = single(fread(fid_in, npol*Nin*NperNin, ntype));

    if feof(fid_in)
        error('Error - hit end of input file!');
    end;

    %====================
    % Parse real and imag components if incoming data is complex
    switch dformat
        case 'complextocomplex'
            Vstream = reshape(Vstream, 2, []);
            Vstream = complex(Vstream(1,:), Vstream(2,:));
    end;

    % Separate data into different polarisations: Vdat(1,:) and Vdat(2,:)
    Vdat = reshape(Vstream, npol, []);

    %figure;
    %subplot(221); plot((1:Nout),real(Vdat(1,1:Nout))); box on; grid on;
    %title('v1 Real');
    %subplot(223); plot((1:Nout),imag(Vdat(1,1:Nout))); box on; grid on;
    %title('v1 Imag'); xlabel('time');
    %subplot(222); plot((1:Nout),real(Vdat(2,1:Nout))); box on; grid on;
    %title('v2 Real');
    %subplot(224); plot((1:Nout),imag(Vdat(2,1:Nout))); box on; grid on;
    %title('v2 Imag'); xlabel('time');
    %pause

    % Evaluate the channel outputs
    % First pol
    for n = 1 : Nin/M
        if pfb_type == 0,
            y2(1,:,n) = CS_PFB_1(Vdat(1,(n-1)*L+1:n*L));
        else
            y2(1,:,n) = OS_PFB_1(Vdat(1,(n-1)*M+1:1:n*M));
        end;
    end;
    % Second pol - must use different function due to persistent variables
    for n = 1 : Nin/M
        if pfb_type == 0,
            y2(2,:,n) = CS_PFB_2(Vdat(2,(n-1)*L+1:n*L));
        else
            y2(2,:,n) = OS_PFB_2(Vdat(2,(n-1)*M+1:1:n*M));
        end;
    end;

    %y2_plot(1:Nin/L) = y2(1,3,(1:Nin/L));
    %subplot(211); plot((1:Nin/L),real(y2_plot(1:Nin/L))); box on; grid on;
    %title('Output Real');
    %subplot(212); plot((1:Nin/L),imag(y2_plot(1:Nin/L))); box on; grid on;
    %title('Output Imag'); xlabel('time');
    %pause

    % Interleave polarizations and real/imag
    % (selecting just the required output channel number)
    z1_y2(1:Nin/M) = y2(1,chan_no,(1:Nin/M));
    z2_y2(1:Nin/M) = y2(2,chan_no,(1:Nin/M));
    z = [real(transpose(z1_y2)), imag(transpose(z1_y2)),...
         real(transpose(z2_y2)), imag(transpose(z2_y2))];
    dat = reshape(transpose(z),2*npol*Nin/M,1);

    %Write vector to file
    fwrite(fid_out, dat, ntype);
end;

fclose(fid_in);
fclose(fid_out);


return

exit();
end



% First CS-PFB
% Critically sampled Polyphase Filter-Bank Channelizer function, based on
% code by Thushara Kanchana Gunaratne, RO/RCO, NSI-NRC, Canada, 2015-03-05
function y = CS_PFB_1(x)

global L; global fname_pfb;

%Declaration and Initialization of Input Mask
%As Persistence Variables
persistent n h xM;
if isempty(n)

    %Loading the Prototype Filter as an initiation task
    %This Will NOT repeat in subsequent runs
    FiltCoefStruct = load(fname_pfb);
    h = FiltCoefStruct.h;

    %Initiate the Input Mask that is multiplied with the Filter mask
    xM = zeros(1,length(h));
    %Initiate the Output mask
    yP = zeros(L,1);

    %Control Index - Initiation
    n = 0;

end; %End if

%Multiplying the Indexed Input Mask and Filter Mask elements and
%accumulating
for k = 1 : L
    yP(k,1) = sum(xM(k:L:end).*h(k:L:end));
end; % For k

%The Linear Shift of Input through the FIFO
%Shift the Current Samples by M to the Right
xM(1,L+1:end) = xM(1,1:end-L);
%Assign the New Input Samples for the first M samples
xM(1,1:L) = fliplr(x);%Note the Flip (Left-Right) place the Newest sample
                      % to the front

%transpose(yP((1:L),1))

% %Evaluating the Cross-Stream (i.e. column wise) IDFT
% yfft = L*L*(ifft(yP));%
%
% %Note the Input Signal is Real-Valued. Hence, only half of the output
% %Channels are Independent. The Packing Method is used here. However,
% %any Optimized Real IFFT Evaluation Algorithm Can be used in its place
% %Evaluating the Cross-Stream (i.e. column wise) IDFT using Packing
% %Method
% %The Complex-Valued Sequence of Half Size
y2C = yP(1:2:end) + 1j*yP(2:2:end);
%The Complex IDFT of LC=L/2 Points
IFY2C = L*L/2*ifft(y2C);
%
y(1:L/2) = (0.5*((IFY2C+conj(circshift(flipud(IFY2C),[+1,0])))...
            - 1j*exp(2j*pi*(0:1:L/2-1).'/L).*...
              (IFY2C-conj(circshift(flipud(IFY2C),[+1,0])))));
% [0,+1]
y(L/2+1) = 0.5*((IFY2C(1)+conj(IFY2C(1)) + 1j*(IFY2C(1)-conj(IFY2C(1)))));

y(L/2+2:L) = conj(fliplr(y(2:L/2)));

%Changing the Control Index
n = n+1;

end %Function CS_PFB_1



% Second CS-PFB
% Critically sampled Polyphase Filter-Bank Channelizer function, based on
% code by Thushara Kanchana Gunaratne, RO/RCO, NSI-NRC, Canada, 2015-03-05
function y = CS_PFB_2(x)

global L; global fname_pfb;

%Declaration and Initialization of Input Mask
%As Persistence Variables
persistent n h xM;
if isempty(n)

    %Loading the Prototype Filter as an initiation task
    %This Will NOT repeat in subsequent runs
    FiltCoefStruct = load(fname_pfb);
    h = FiltCoefStruct.h;

    %Initiate the Input Mask that is multiplied with the Filter mask
    xM = zeros(1,length(h));
    %Initiate the Output mask
    yP = zeros(L,1);

    %Control Index - Initiation
    n = 0;

end; %End if

%Multiplying the Indexed Input Mask and Filter Mask elements and
%accumulating
for k = 1 : L
    yP(k,1) = sum(xM(k:L:end).*h(k:L:end));
end; % For k

%The Linear Shift of Input through the FIFO
%Shift the Current Samples by M to the Right
xM(1,L+1:end) = xM(1,1:end-L);
%Assign the New Input Samples for the first M samples
xM(1,1:L) = fliplr(x);%Note the Flip (Left-Right) place the Newest sample
                      % to the front

%transpose(yP((1:L),1))

% %Evaluating the Cross-Stream (i.e. column wise) IDFT
% yfft = L*L*(ifft(yP));%
%
% %Note the Input Signal is Real-Valued. Hence, only half of the output
% %Channels are Independent. The Packing Method is used here. However,
% %any Optimized Real IFFT Evaluation Algorithm Can be used in its place
% %Evaluating the Cross-Stream (i.e. column wise) IDFT using Packing
% %Method
% %The Complex-Valued Sequence of Half Size
y2C = yP(1:2:end) + 1j*yP(2:2:end);
%The Complex IDFT of LC=L/2 Points
IFY2C = L*L/2*ifft(y2C);
%
y(1:L/2) = (0.5*((IFY2C+conj(circshift(flipud(IFY2C),[+1,0])))...
            - 1j*exp(2j*pi*(0:1:L/2-1).'/L).*...
              (IFY2C-conj(circshift(flipud(IFY2C),[+1,0])))));
% [0,+1]
y(L/2+1) = 0.5*((IFY2C(1)+conj(IFY2C(1)) + 1j*(IFY2C(1)-conj(IFY2C(1)))));

y(L/2+2:L) = conj(fliplr(y(2:L/2)));

%Changing the Control Index
n = n+1;

end %Function CS_PFB_2




% First OS-PFB
% Oversampled Polyphase Filter-Bank Channelizer function, based on code by
% Thushara Kanchana Gunaratne, RO/RCO, NSI-NRC, Canada, 2015-03-05
function y = OS_PFB_1(x)

global L; global Nu; global M; global L_M; global fname_pfb;

%Declaration and Initialization of Input Mask
%As Persistance Variables
persistent n h xM;
if isempty(n)

    %Loading the Prototype Filter as an initiation task
    %This Will NOT repeat in subsequent runs
    FiltCoefStruct = load(fname_pfb);
    h = FiltCoefStruct.h;

    %Initiate the Input Mask that is multiplied with the Filter mask
    xM = zeros(1,length(h));
    %Initiate the Output mask
    yP = zeros(L,1);

    %Control Index - Initiation
    n = 0;

end; %End if

%Multiplying the Indexed Input Mask and Filter Mask elements and
%accumulating
for k = 1 : L
    yP(k,1) = sum(xM(k:L:end).*h(k:L:end));
end; % For k

%The Linear Shift of Input through the FIFO
%Shift the Current Samples by M to the Right
xM(1,M+1:end) = xM(1,1:end-M);
%Assign the New Input Samples for the first M samples
xM(1,1:M) = fliplr(x);%Note the Flip (Left-Right) place the Newest sample
                      % to the front

%Performing the Circular Shift to Compensate the Shift in Band Center
%Frequencies
if n == 0
    y1S = yP;
else
    y1S = [yP((Nu-n)*L_M+1:end); yP(1:(Nu-n)*L_M)];
end;

% %Evaluating the Cross-Stream (i.e. column wise) IDFT
% yfft = L*L*(ifft(yP));%
%
% %Modulating the Channels (i.e. FFT Outputs) to compensate the shift in the
% %center frequency
% %y = yfft.*exp(2j*pi*(1-M/L)*n*(0:1:L-1).');
% y = yfft.*exp(-2j*pi*M/L*n*(0:1:L-1).');

% %Note the Input Signal is Real-Valued. Hence, only half of the output
% %Channels are Independent. The Packing Method is used here. However,
% %any Optimized Real IFFT Evaluation Algorithm Can be used in its place
% %Evaluating the Cross-Stream (i.e. column wise) IDFT using Packing
% %Method
% %The Complex-Valued Sequence of Half Size
y2C = y1S(1:2:end) + 1j*y1S(2:2:end);
%The Complex IDFT of LC=L/2 Points
IFY2C = L*L/2*ifft(y2C);
%
y(1:L/2) = (0.5*((IFY2C+conj(circshift(flipud(IFY2C),[+1,0])))...
            - 1j*exp(2j*pi*(0:1:L/2-1).'/L).*...
             (IFY2C-conj(circshift(flipud(IFY2C),[+1,0])))));
% [0,+1]
y(L/2+1) = 0.5*((IFY2C(1)+conj(IFY2C(1)) + 1j*(IFY2C(1)-conj(IFY2C(1)))));

y(L/2+2:L) = conj(fliplr(y(2:L/2)));

%Changing the Control Index
n = n+1;
n = mod(n,Nu);

end %Function OS_PFB_1



% Second OS-PFB
% Oversampled Polyphase Filter-Bank Channelizer function, based on code by
% Thushara Kanchana Gunaratne, RO/RCO, NSI-NRC, Canada, 2015-03-05
function y = OS_PFB_2(x)

global L; global Nu; global M; global L_M; global fname_pfb;

%Declaration and Initialization of Input Mask
%As Persistance Variables
persistent n h xM;
if isempty(n)

    %Loading the Prototype Filter as an initiation task
    %This Will NOT repeat in subsequent runs
    FiltCoefStruct = load(fname_pfb);
    h = FiltCoefStruct.h;

    %Initiate the Input Mask that is multiplied with the Filter mask
    xM = zeros(1,length(h));
    %Initiate the Output mask
    yP = zeros(L,1);

    %Control Index - Initiation
    n = 0;

end; %End if

%Multiplying the Indexed Input Mask and Filter Mask elements and
%accumulating
for k = 1 : L
    yP(k,1) = sum(xM(k:L:end).*h(k:L:end));
end; % For k

%The Linear Shift of Input through the FIFO
%Shift the Current Samples by M to the Right
xM(1,M+1:end) = xM(1,1:end-M);
%Assign the New Input Samples for the first M samples
xM(1,1:M) = fliplr(x);%Note the Flip (Left-Right) place the Newest sample
                      % to the front

%Performing the Circular Shift to Compensate the Shift in Band Center
%Frequencies
if n == 0
    y1S = yP;
else
    y1S = [yP((Nu-n)*L_M+1:end); yP(1:(Nu-n)*L_M)];
end;

% %Evaluating the Cross-Stream (i.e. column wise) IDFT
% yfft = L*L*(ifft(yP));%
%
% %Modulating the Channels (i.e. FFT Outputs) to compensate the shift in the
% %center frequency
% %y = yfft.*exp(2j*pi*(1-M/L)*n*(0:1:L-1).');
% y = yfft.*exp(-2j*pi*M/L*n*(0:1:L-1).');

% %Note the Input Signal is Real-Valued. Hence, only half of the output
% %Channels are Independent. The Packing Method is used here. However,
% %any Optimized Real IFFT Evaluation Algorithm Can be used in its place
% %Evaluating the Cross-Stream (i.e. column wise) IDFT using Packing
% %Method
% %The Complex-Valued Sequence of Half Size
y2C = y1S(1:2:end) + 1j*y1S(2:2:end);
%The Complex IDFT of LC=L/2 Points
IFY2C = L*L/2*ifft(y2C);
%
y(1:L/2) = (0.5*((IFY2C+conj(circshift(flipud(IFY2C),[+1,0])))...
            - 1j*exp(2j*pi*(0:1:L/2-1).'/L).*...
             (IFY2C-conj(circshift(flipud(IFY2C),[+1,0])))));
% [0,+1]
y(L/2+1) = 0.5*((IFY2C(1)+conj(IFY2C(1)) + 1j*(IFY2C(1)-conj(IFY2C(1)))));

y(L/2+2:L) = conj(fliplr(y(2:L/2)));

%Changing the Control Index
n = n+1;
n = mod(n,Nu);

end %Function OS_PFB_2


% Function to pull observation parameters from a header file
% Also adjusts the TSAMP value to account for Over-Sampling and writes a new header
function headerMap = headerReadWrite(headerFile, fname_out, headerMap)

if exist(headerFile, 'file')
    fInputHeaderFile = fopen(headerFile, 'r');
    fid_out = fopen(fname_out, 'a');
    formatSpec = '%c'; %collects all chars
    headerString = fscanf(fInputHeaderFile, formatSpec);
    headerLines = strsplit(headerString, '\n');

    for i=1:length(headerLines)
        % Map parameter names to values
        tempMap = strsplit(headerLines{i}); % Parse lines along whitespace

        % Only consider meaningful lines
        if length(tempMap) > 1
            headerMap(tempMap{1}) = tempMap{2};

            % TSAMP must be rescaled before appending
            new_line = strcat(headerLines{i},'\n');
            if strcmp(tempMap{1},'TSAMP')
                % Default, if any of TSAMP, NCHAN, OS_FACTOR don't exist
                tsamp_line = new_line;
            else % All lines that are not TSAMP
                fprintf(fid_out, new_line);
            end
        end
    end

    % Get TSAMP, NCHAN, OS_FACTOR as numbers
    if isKey(headerMap,'TSAMP')
        tsamp_val = str2num(headerMap('TSAMP'));

        % Defaults
        nchan_val = 8;
            % PFB downsamples by 8, only outputs 1 of the channels,
            % DSPSR only sees 1 channel, so we can't use NCHAN in the header
        Nu_val = 1;
        De_val = 1;

        % Number of Channels
        if isKey(headerMap,'NCHAN')
            nchan_val = str2num(headerMap('NCHAN'));
        end

        % Over-Sampling Factor
        if isKey(headerMap,'OS_FACTOR')
            splitInput = strsplit(headerMap('OS_FACTOR'),'/');
            Nu_val = str2num(splitInput{1});
            De_val = str2num(splitInput{2});
        end

        % Fix TSAMP and append
        digitsOld = digits(10); %Increase precision to 10 digits
        tsamp_val = vpa(tsamp_val*nchan_val*De_val/Nu_val); % TSAMP*NCHAN/OS_FACTOR
        tsamp_line = ['TSAMP' '        ' char(tsamp_val) '\n'];
    end

    % Append TSAMP onto output header
    fprintf(fid_out, tsamp_line);

    fclose(fInputHeaderFile);
    fclose(fid_out);
end

return
end % Function headerReadWrite
