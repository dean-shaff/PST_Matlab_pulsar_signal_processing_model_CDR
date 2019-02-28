dspsr -c 0.00575745 -D 2.64476 -IF 1:D  data/py_channelized.simulated_pulsar.noise_0.0.nseries_10.ndim_2.os.dump \
  -O py.inverse.pulsar.10.os -V -dump Detection 2> inverse.os.log \
  && mv pre_Detection.dump pre_Detection.py.inverse.pulsar.10.os.dump
# dspsr -c 0.00575745 -D 2.64476  data/simulated_pulsar.noise_0.0.nseries_10.ndim_2.dump \
#   -O pulsar.10 -dump Detection && mv pre_Detection.dump pre_Detection.vanilla.pulsar.10.dump
# pipenv run python validate_dspsr_pfb_inversion.py
