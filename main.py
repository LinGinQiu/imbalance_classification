import os

import numpy as np
import pandas as pd
import data_struc
from config import Config
import logging
from utils import *
import time
from numpy import interp
from aeon.datasets import load_from_tsv_file, write_to_tsfile, load_from_tsfile
from run_loop import main_loop
import argparse

# Configure logging to write to a file
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--run", action="store_true", help="Choose to run or debug (in local) the script")

args = parser.parse_args()
config = Config(args.run)


if __name__ == "__main__":
    if args.run:
        print("Running the script")
        for classifier in config.classification_methods:
            config.oversampling_methods = config.oversampling
            config.classifier = classifier
            config.check_path()
            main_loop(args, config, config.classifier)
    else:
        print("Debugging the script")
        main_loop(args, config, config.classifier)
