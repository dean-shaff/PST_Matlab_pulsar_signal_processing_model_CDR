function inversion_pipeline()

  % simulated_pulsar_filename = 'data/simulated_pulsar.noise_0.0.nseries_5.ndim_2.dump';
  % channelized_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_2.cs.dump';
  % inverted_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_2.cs.dump.inverted.mat';

  simulated_pulsar_filename = 'data/simulated_pulsar.noise_0.0.nseries_5.ndim_2.dump';
  channelized_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_2.os.dump';
  inverted_filename_matlab = 'data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_2.os.dump.inverted.mat';
  inverted_filename_dspsr = 'pre_Detection.dump';

  %  the following takes a long time -- Only uncomment if you need to regenerated PFB channelization.
  % PFBchannelizer(simulated_pulsar_filename, -1, '8/7');

  % PFBinversion(channelized_filename, 1, 1);
  %
  % compare_inversion(simulated_pulsar_filename, inverted_filename_matlab, 0, 1, 0);
  % compare_inversion(simulated_pulsar_filename, inverted_filename_dspsr, 0, 1, 0);

  compare_inversion_techniques(inverted_filename_matlab, inverted_filename_dspsr);

end
