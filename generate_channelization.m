signal_file = "simulated_pulsar.dump"
if ~ exist(signal_file, "file")
    fprintf("signal file doesn't exist. Generating...\n") 
    signalgen()
else
    fprintf("signal file already exists\n")
end

fprintf("running channelizer\n")

PFBchannelizer()



