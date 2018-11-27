#!/bin/bash

function main {
  local DM=${1}
  local dspsr_DM=${2}
  if [ "${dspsr_DM}" == "" ]; then
    dspsr_DM="1"
  fi
  pulse_period="0.00575745"
  # products_file="products/dm.${DM}"
  # dspsr_cmd="dspsr -F 128:D -c ${pulse_period} -D ${dspsr_DM} data/simulated_pulsar.dm.${DM}.dump -O ${products_file}"
  products_file="products/inverse"
  dspsr_cmd="dspsr -IF -c ${pulse_period} -D ${dspsr_DM} data/full_channelized_pulsar.dump -O ${products_file}"
  psrplot_cmd="psrplot -p freq+ ${products_file}.ar -jD"
  echo ${dspsr_cmd}
  dspsr_out=$(eval ${dspsr_cmd})
  echo ${psrplot_cmd}
  psrplot_out=$(eval ${psrplot_cmd})
}

main $@
