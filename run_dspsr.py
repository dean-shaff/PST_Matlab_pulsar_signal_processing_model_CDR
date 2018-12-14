#! ~/linux_64/bin/python3
import os
import shlex
import argparse
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
product_dir = os.path.join(current_dir, "products")
log_dir = os.path.join(current_dir, "logs")

pulsar_period = 0.00575745
dm = 2.64476
pfb_types = ["os", "cs"]

def create_parser():

    parser = argparse.ArgumentParser(description="Run dspsr")
    parser.add_argument("-pfb", dest="pfb", default="os")
    parser.add_argument("-nf", dest="nf", default="0.0")
    parser.add_argument("-ns", dest="ns", default="2")
    parser.add_argument("--input_dim", dest="input_dim", default="2")
    parser.add_argument("-f", dest="forward", action="store_true")
    return parser

def main():
    parsed = create_parser().parse_args()
    assert(parsed.pfb in pfb_types)
    data_file_path = os.path.join(
        data_dir,
        "full_channelized_pulsar.noise_{}.nseries_{}.ndim_{}.{}.dump".format(
            parsed.nf,
            parsed.ns,
            parsed.input_dim,
            parsed.pfb
        )
    )
    product_file_path = os.path.join(
        product_dir,
        "inverse.noise_{}.nseries_{}.ndim_{}.{}".format(
            parsed.nf,
            parsed.ns,
            parsed.input_dim,
            parsed.pfb
        )
    )
    log_file_path = product_file_path.replace(product_dir, log_dir) + ".log"

    flags = "-IF -F 1:D"

    cmd = f"dspsr {flags} -c {pulsar_period} -D {dm} {data_file_path} -O {product_file_path} -V"
    print(f"Executing command {cmd}")
    log_file = open(log_file_path, "w")
    p = subprocess.run(shlex.split(cmd), stderr=log_file)
    log_file.close()

if __name__ == "__main__":
    main()
