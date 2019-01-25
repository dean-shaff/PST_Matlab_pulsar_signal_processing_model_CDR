function inversion_pipeline()

  signalgen(3, 0);
  close all;
  simulated_pulsar_filename = 'data/simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump';
  % channelized_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_3.ndim_2.cs.dump';
  % inverted_filename_matlab = 'data/full_channelized_pulsar.noise_0.0.nseries_3.ndim_2.cs.dump.inverted.mat';
  PFBchannelizer(simulated_pulsar_filename, -1, '1/1');

  % channelized_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_3.ndim_2.os.dump';
  % inverted_filename_matlab = 'data/full_channelized_pulsar.noise_0.0.nseries_3.ndim_2.os.dump.inverted.mat';
  PFBchannelizer(simulated_pulsar_filename, -1, '8/7');

  % inverted_filename_dspsr = 'pre_Detection.dump';

  % inverted_filename_matlab_1 = PFBinversion(channelized_filename, 1, 1);
  % inverted_filename_matlab_2 = PFBinversion(channelized_filename, 1, 2);
  % inverted_filename_matlab_3 = PFBinversion(channelized_filename, 1, 3);

  % compare_inversion(simulated_pulsar_filename, inverted_filename_matlab_2, 0, 1, 0);
  % compare_inversion(simulated_pulsar_filename, inverted_filename_matlab_3, 0, 1, 0);
  % compare_inversion(simulated_pulsar_filename, inverted_filename_dspsr, 0, 1, 0);

  % compare_inversion_techniques(inverted_filename_matlab_2, inverted_filename_dspsr);
  % compare_inversion_techniques(inverted_filename_matlab_1, inverted_filename_matlab_3);
  % compare_inversion_techniques(inverted_filename_matlab_1, inverted_filename_matlab_2);
  % compare_inversion_techniques(inverted_filename_matlab_2, inverted_filename_matlab_3);

end
