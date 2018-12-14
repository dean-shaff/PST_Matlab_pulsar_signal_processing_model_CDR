using dsp::LoadToFold
opening data file /home/SWIN/dshaff/mnt/ozstar/projects/PST_Matlab_pulsar_signal_processing_model_CDR/data/full_channelized_pulsar.noise_0.0.nseries_5.os.dump
dsp::File::create filename='/home/SWIN/dshaff/mnt/ozstar/projects/PST_Matlab_pulsar_signal_processing_model_CDR/data/full_channelized_pulsar.noise_0.0.nseries_5.os.dump
dsp::File::create with 22 registered sub-classes
dsp::File::create testing Dummy
dsp::DummyFile::is_valid first line != DUMMY
dsp::File::create testing DADA
dsp::File::create DADA::is_valid() returned true
dsp::Input ctor this=0x1eda670
dsp::File::open filename=/home/SWIN/dshaff/mnt/ozstar/projects/PST_Matlab_pulsar_signal_processing_model_CDR/data/full_channelized_pulsar.noise_0.0.nseries_5.os.dump
ASCIIObservation::load required keywords:
  TELESCOPE
  SOURCE
  CALFREQ
  FREQ
  BW
  NPOL
  NBIT
  TSAMP
  UTC_START
  OBS_OFFSET
dsp::ASCIIObservation::load ndat=0
dsp::ASCIIObservation::load UTC_START='2018-12-13-00:28:18'
dsp::ASCIIObservation::load asctime=Thu Dec 13 00:28:18 2018

dsp::ASCIIObservation::load start_mjd=58465
DADAFile::open exit
dsp::File::fstat_file_ndat(): buf=90181632 header_bytes=4096 tailer_bytes=0 total_bytes=90177536
dsp::File::seek_bytes nbytes=0 header_bytes=4096
dsp::Input::seek [INTERNAL] resolution=1 resolution_offset=0 load_sample=0
dsp::Input::set_load_size block_size=0 resolution_offset=0
dsp::Input::set_load_size load_size=0
dsp::SingleThread::set_input input=0x1eda670
dsp::Unpacker::create with 26 registered sub-classes
dsp::Unpacker::create testing FloatUnpacker
dsp::Transformation[FloatUnpacker]::ctor
dsp::FloatUnpacker ctor
dsp::Unpacker::match
dsp::Unpacker::create return new sub-class
    dsp::LoadToFold::construct
Creating WeightedTimeSeries instance
dsp::IOManager::set_output (TimeSeries*) 0x1edbf10
dsp::IOManager::set_output call Unpacker::set_output
dsp::Transformation[FloatUnpacker]::set_output (0x1edbf10)
    dsp::Filterbank::construct: setting up filterbank
Creating WeightedTimeSeries instance
dsp::Transformation[Filterbank]::ctor
dsp::Filterbank::ctor
dsp::Filterbank::set_engine(0x1ed4d10)
dsp::Transformation[Filterbank]::set_input (0x1edbf10)
dsp::Transformation[Filterbank]::set_output (0x1ed8d80)
dsp::Transformation[Detection]::ctor
dsp::LoadToFold::construct: created Detection object
dsp::Transformation[Detection]::set_input (0x1ed8d80)
dsp::Transformation[Detection]::set_output (0x1ed8d80)
dsp::Detection::set_output_state to Coherency Products
dsp::LoadToFold::build_fold
dsp::LoadToFold::build_fold nfold=1
dsp::LoadToFold::get_unloader prepare new Archiver
dsp::LoadToFold::build_fold input ptr=0 unloader ptr=0x1eda080
dsp::LoadToFold::build_fold prepare Fold
dsp::Transformation[Fold]::ctor
dsp::PhaseSeries::init this=0x1ed6570
dsp::Transformation[Fold]::set_output (0x1ed6570)
dsp::LoadToFold::build_fold configuring
dsp::LoadToFold::build_fold output ptr=0x1ed6160
dsp::Fold::set_input (TimeSeries* =0x1ed8d80)
dsp::Transformation[Fold]::set_input (0x1ed8d80)
dsp::Fold::set_input input is a WeightedTimeSeries
dsp::Fold::prepare
dsp::Fold::prepare using folding_period=0.00575745
dsp::Fold::reset
PhaseSeries::zero
PhaseSeries::set_hits(0)
PhaseSeries::zero exit
LoadToFold::prepare config DM=2.64476
dsp::IOManager::prepare
dsp::IOManager::set_output (BitSeries*) 0x1ed6a50
dsp::IOManager::set_output call Unpacker::set_input
dsp::Transformation[FloatUnpacker]::set_input (0x1ed6a50)
dsp::Input::set_output (BitSeries* = 0x1ed6a50)
dsp::Input::prepare
dsp::BitSeries::copy_configuration ndat=704512
dsp::Input::prepare output start_time=58465
dsp::Unpacker::prepare
dsp::TimeSeries::set_nbit (32) ignored
dsp::Unpacker::prepare output start_time=58465
dsp::Filterbank::prepare: prepared: 0
dsp::Filterbank::_preparationforDataProcessing
dsp::Filterbank::_computeSampleCounts: nchan: 1
dsp::Filterbank::_computeSampleCounts: _n_input_chan: 8
dsp::Dedispersion::match before lock
dsp::Dedispersion::match after lock
dsp::Dedispersion::prepare input.nchan=8 channels=1
	 centre frequency=1405 bandwidth=80 dispersion measure=2.64476
dsp::Dedispersion::smearing_time freq=1365 bw=40
dsp::Dedispersion::delay_time DM=2.64476 f1=1345 f2=1385
dsp::Dedispersion::smearing_time in the lower half of the band: 0.345341 ms
dsp::Dedispersion::smearing_samples = 27627
dsp::Dedispersion::smearing_samples effective smear time: 0.379875 ms (30390 pts).
dsp::Dedispersion::prepare 0 unsupported channels
dsp::Dedispersion::smearing_time freq=1445 bw=40
dsp::Dedispersion::delay_time DM=2.64476 f1=1425 f2=1465
dsp::Dedispersion::smearing_time in the upper half of the band: 0.291087 ms
dsp::Dedispersion::smearing_samples = 23286
dsp::Dedispersion::smearing_samples effective smear time: 0.3202 ms (25616 pts).
dsp::Response::get_minimum_ndat impulse_tot=56006 min power of two=65536
Response::set_optimal_ndat minimum ndat=65536
optimal_fft_length: minimum FFT:65536 maximum FFT:0
NFFT 65536  %kept:0.1454 O(FFT):726817.4980 Timescale:76.2663
NFFT 131072  %kept:0.5727 O(FFT):1544487.1833 Timescale:20.5751
NFFT 262144  %kept:0.7864 O(FFT):3270678.7410 Timescale:15.8665
NFFT 524288  %kept:0.8932 O(FFT):6904766.2310 Timescale:14.7449
NFFT 1048576  %kept:0.9466 O(FFT):14536349.9601 Timescale:14.6452
NFFT 2097152  %kept:0.9733 O(FFT):30526334.9161 Timescale:14.9555
dsp::Dedispersion::resize: _npol: 2 _nchan: 1 _ndat: 1048576 _ndim: 1
dsp::Shape::resize  npol=2  nchan=1  ndat=1048576  ndim=1
dsp::Dedispersion::build
  centre frequency = 1405
  bandwidth = 80
  dispersion measure = 2.64476
  Doppler shift = 1
  ndat = 1048576
  nchan = 1
  centred on DC = 0
  fractional delay compensation = 0
dsp::Dedispersion::resize: _npol: 1 _nchan: 1 _ndat: 1048576 _ndim: 2
dsp::Shape::resize  npol=1  nchan=1  ndat=1048576  ndim=2
dsp::Response::match input.nchan=8 channels=1
dsp::Response::match swap channels (nchan=8)
dsp::Dedispersion::match exit
dsp::Filterbank::_computeSampleCounts: _input_fft_length: 131072
dsp::Filterbank::_computeSampleCounts: _output_fft_length: 1048576
dsp::Filterbank::_computeSampleCounts: _input_discard: 3202 3799
dsp::Filterbank::_computeSampleCounts: _output_discard: 25616 30392
dsp::Filterbank::_computeScaleFactor
dsp::Filterbank::_computeScaleFactor: scalefac: 1.09951e+12
dsp::Filterbank::_setMinimumSamples
dsp::Filterbank::_setMinimumSamples: min_samples: 1048576
dsp::Reserve::reserve increasing reserve to 1048576
dsp::TimeSeries::change_reserve (1048576)
dsp::Filterbank::_prepareOutput: ndat: 0 set_ndat: 0
dsp::Filterbank::_configOutputStructure: ndat: 0 set_ndat: 0
dsp::TimeSeries::set_nbit (32) ignored
dsp::TimeSeries::copy_configuration ndat=0
dsp::WeightedTimeSeries::copy_weights resize weights (ndat=0)
dsp::WeightedTimeSeries::resize_weights nsamples=0
dsp::WeightedTimeSeries::resize_weights reserve=0
dsp::WeightedTimeSeries::resize_weights nweights=0 require=0 have=0
dsp::Filterbank::_configOutputStructure tres_ratio: 1
dsp::Filterbank::_configWeightedOutput: tres_ratio: 1
dsp::WeightedTimeSeries::convolve_weights ndat=0 + nkeep=124071 is less than nfft=131072
dsp::WeightedTimeSeries::get_nweights weight_idat=0 nweights=0
dsp::WeightedTimeSeries::scrunch_weights nscrunch=1 ndat_per_weight=0 nweights=0 weight_idat=0
dsp::Filterbank::_configOutputStructure scrunch ndat: 0
dsp::WeightedTimeSeries::resize (0)
dsp::TimeSeries::resize (0) data=0 buffer=0 ndat=0
dsp::TimeSeries::resize reserve_ndat=0 fake_ndat=0
dsp::WeightedTimeSeries::resize_weights nsamples=0
dsp::WeightedTimeSeries::resize_weights reserve=0
dsp::WeightedTimeSeries::resize_weights nweights=0 require=0 have=0
dsp::Filterbank::_configOutputStructure output nchan: 1
dsp::Filterbank::_configOutputStructure output npol: 2
dsp::Filterbank::_configOutputStructure output ndim: 2
dsp::Filterbank::_configOutputStructure output ndat: 0 scale: 1.09951e+12
dsp::Filterbank::_prepareOutput: input rate: 1.14286e+07 output rate: 9.14286e+07
dsp::Filterbank::_prepareOutput: input->get_dual_sideband() 1
dsp::Filterbank::_prepareOutput: setting output->set_swap(true)
dsp::Filterbank::_prepareOutput start time += 25616 samps -> 58465
dsp::Dedispersion::mark no longer changing DM
Filterbank::_prepareOutput: done
dsp::Filterbank::_setupEngine
dsp::FilterbankInverseEngineCPU::setup: nChannels: 1
dsp::FilterbankInverseEngineCPU::setup: nData: 1048576
dsp::FilterbankInverseEngineCPU::setup: n_dims: 2
dsp::FilterbankInverseEngineCPU::setup: nFilterPositive: 25616
dsp::FilterbankInverseEngineCPU::setup: nFilterNegative: 30390
dsp::FilterbankInverseEngineCPU::setup: nFilterTotal: 56006
dsp::FilterbankInverseEngineCPU::setup: filterbank frequency resolution: 1048576
dsp::FilterbankInverseEngineCPU::setup: _scaleFactor: 1.37439e+11
dsp::FilterbankInverseEngineCPU::setup: _nSampleOverlap: 7000
dsp::FilterbankInverseEngineCPU::setup: _input_fft_length: 131072
dsp::FilterbankInverseEngineCPU::setup: _nSampleStep: 124072
dsp::Filterbank::_preparationforDataProcessing: done
dsp::Detection::resize_output
dsp::Detection::resize_output state: Coherency Products ndim=4
dsp::Detection::resize_output reshape FROM npol=2 ndim=2 TO npol=1 ndim=4
dsp::Fold::prepare
dsp::Fold::prepare using folding_period=0.00575745
dsp::Fold::reset
PhaseSeries::zero
PhaseSeries::set_hits(0)
PhaseSeries::zero exit
dsp::Response::get_minimum_ndat impulse_tot=56006 min power of two=65536
dspsr: dedispersion filter length=1048576 (minimum=65536) complex samples
dspsr: 1 channel dedispersing filterbank requires 1048576 samples
dsp::IOManager::set_overlap request overlap=0
dsp::IOManager::set_overlap input resolution=1
dsp::IOManager::set_overlap require overlap=0
dsp::IOManager::set_block_size minimum_samples=1048576
dsp::IOManager::set_block_size input resolution=1
dsp::IOManager::set_block_size copies=2 nbit=32 nbyte=12
dsp::IOManager::set_block_size required block_size=1048576
dsp::IOManager::set_block_size minimum_RAM=0 nbyte_dat=384
dsp::IOManager::set_block_size maximum block_size=699048
dsp::IOManager::set_block_size insufficient RAM
dsp::Transformation[Filterbank]::dtor
dsp::Transformation[Detection]::dtor
dsp::Transformation[Fold]::dtor
dsp::PhaseSeries::~PhaseSeries this=0x1ed6570 hits=0
dsp::DataSeries::resize nsamp=0 nbit=32 ndim=1 (current ndat=0)
dsp::DataSeries::resize nbits=nsamp*nbit*ndim=0
dsp::DataSeries::resize npol=1 nchan=1
dsp::DataSeries::resize nbytes=nbits/8*npol*nchan=0 (current size=0)
dsp::DataSeries::resize nsamp=0 nbit=32 ndim=4 (current ndat=0)
dsp::DataSeries::resize nbits=nsamp*nbit*ndim=0
dsp::DataSeries::resize npol=1 nchan=1
dsp::DataSeries::resize nbytes=nbits/8*npol*nchan=0 (current size=0)
dsp::Transformation[FloatUnpacker]::dtor
dsp::DataSeries::resize nsamp=0 nbit=32 ndim=2 (current ndat=0)
dsp::DataSeries::resize nbits=nsamp*nbit*ndim=0
dsp::DataSeries::resize npol=2 nchan=8
dsp::DataSeries::resize nbytes=nbits/8*npol*nchan=0 (current size=0)

Error::stack
	dsp::IOManager::set_block_size
Error::InvalidState
Error::message
	insufficient RAM: limit=256 MB -> block=699048 samples
	require=1048576 samples -> "-U 384" on command line

Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
Pulsar::Archive::Agent::~Agent
dsp::Transformation[GenericFourBitUnpacker]::dtor
dsp::Transformation[GenericEightBitUnpacker]::dtor
dsp::Transformation[SigProcUnpacker]::dtor
dsp::Transformation[S2TwoBitCorrection]::dtor
dsp::Transformation[PuMa2Unpacker]::dtor
dsp::Transformation[OneBitCorrection]::dtor
dsp::Transformation[EDAFourBit]::dtor
dsp::Transformation[MaximUnpacker]::dtor
dsp::Transformation[Mark5TwoBitCorrection]::dtor
dsp::Transformation[Mark5Unpacker]::dtor
dsp::Transformation[Mark4TwoBitCorrection]::dtor
dsp::Transformation[LBADR64_TwoBitCorrection]::dtor
dsp::Transformation[SMROTwoBitCorrection]::dtor
dsp::Transformation[GMRTFilterbank16]::dtor
dsp::Transformation[GMRTFourBit]::dtor
dsp::Transformation[GMRTEightBit]::dtor
dsp::Transformation[CPSR2TwoBitCorrection]::dtor
dsp::Transformation[CPSRTwoBitCorrection]::dtor
dsp::Transformation[BPSRCrossUnpacker]::dtor
dsp::Transformation[BPSRUnpacker]::dtor
dsp::Transformation[BCPMUnpacker]::dtor
dsp::Transformation[ASPUnpacker]::dtor
dsp::Transformation[APSREightBit]::dtor
dsp::Transformation[APSRFourBit]::dtor
dsp::Transformation[APSRTwoBitCorrection]::dtor
dsp::Transformation[FloatUnpacker]::dtor
