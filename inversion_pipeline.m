function inversion_pipeline()
  simulated_pulsar_filename = 'data/simulated_pulsar.noise_0.0.nseries_5.ndim_2.dump';
  channelized_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_2.cs.dump';
  inverted_filename = 'data/full_channelized_pulsar.noise_0.0.nseries_5.ndim_2.cs.dump.inverted.mat';

  PFBinversion(channelized_filename);

  compare_inversion(simulated_pulsar_filename, inverted_filename, 0, 1, 1)

end
