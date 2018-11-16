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
% Ignores the header section.
%
% Changes:
% --------
%
% Author           Date         Comments
% ---------------  -----------  ----------------------------------------
% I. Morrison      31-Jul-2015  Original version
%
% ----------------------------------------------------------------------
 
close all; clear all; clc;

% Input file name
fname_in = 'simulated_pulsar.dump';

% Output file name
%fname_out = 'cs_channelized_pulsar.dump';
fname_out = 'os_channelized_pulsar.dump';

%=======================================
% PFB parameters

% Define globals common also to CS_PFB() / OS_PFB() sub-functions
global L; global Nu; global M; global L_M; global fname_pfb;

% Number of channels in filter-bank
L= 8;

% PFB type
pfb_type = 1; % 0 for critically sampled, 1 for oversampled

% OverSampling
Nu = 8; %Numerator
De = 7; %Denominator

if pfb_type == 0,
    Os = 1;
else
    Os = Nu/De; % Oversampling factor
end;
M = L/Os; % Commutator Length
L_M = L-M; % Overlap

% PFB prototype filter coefficients file name
%fname_pfb = 'CS_Prototype_FIR_8.mat';
fname_pfb = 'OS_Prototype_FIR_8.mat';

%=======================================
% Other parameters
hdrsize = 4096; %Header size
hdrtype = 'uint8'; % Data type for header ('uint8' = byte)
ntype = 'single'; % Data type for each element in a pair ('single' = float)
npol = 2; % Number of polarisations (should always be 2 when calc Stokes)
dformat = 'realtocomplex'; %specifies conversion OF real or complex data
%dformat = 'complextocomplex'; %specifies conversion OF real or complex data
Nin = M*(2^14);  % Number of elements per input file read
f_sample_in = 80.; % Sampling frequency of input (MHz)
chan_no = 3; % Particular PFB output channel number to store to file
nseries = 80; % Number of input blocks to read and process

%=============
% Initialisations

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

% Open input file
fid_in = fopen(fname_in);
 
% Read header
fread(fid_in, hdrsize, hdrtype);
%disp(transpose(native2unicode(hdr))); % Show header

% Open file for writing
fid_out = fopen(fname_out, 'w');

% Write header
hdr = zeros(hdrsize,1);
fwrite(fid_out, hdr, hdrtype);

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
%     Nout = Nin/M;
%     figure;
%     subplot(221); plot((1:Nout),real(Vdat(1,1:Nout))); box on; grid on;
%     title('v1 Real'); 
%     subplot(223); plot((1:Nout),imag(Vdat(1,1:Nout))); box on; grid on;
%     title('v1 Imag'); xlabel('time');
%     subplot(222); plot((1:Nout),real(Vdat(2,1:Nout))); box on; grid on;
%     title('v2 Real'); 
%     subplot(224); plot((1:Nout),imag(Vdat(2,1:Nout))); box on; grid on;
%     title('v2 Imag'); xlabel('time');
%     pause

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

%     y2_plot(1:Nin/L) = y2(1,3,(1:Nin/L));
%     subplot(211); plot((1:Nin/L),real(y2_plot(1:Nin/L))); box on; grid on;
%     title('Output Real'); 
%     subplot(212); plot((1:Nin/L),imag(y2_plot(1:Nin/L))); box on; grid on;
%     title('Output Imag'); xlabel('time'); 
%     pause
    
    % Interleave polarizations and real/imag
    % (selecting just the required output channel number)
    z1_y2(1:Nin/M) = y2(1,chan_no,(1:Nin/M));
    z2_y2(1:Nin/M) = y2(2,chan_no,(1:Nin/M));
    z = [real(transpose(z1_y2)), imag(transpose(z1_y2)),...
         real(transpose(z2_y2)), imag(transpose(z2_y2))];
%     x = linspace(0, 1, 16384);
%     pol1 = complex(z(:,1), z(:, 2))
%     if ii == 1 
%        plot(x, abs(pol1).^2); 
%     end
    disp("size(z): ")
    disp(size(z))
    dat = reshape(transpose(z),2*npol*Nin/M,1);
    disp("size(dat): ")
    disp(size(dat))
    %Write vector to file
    fwrite(fid_out, dat, ntype);
end;

fclose(fid_in);
fclose(fid_out);
 
return
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

