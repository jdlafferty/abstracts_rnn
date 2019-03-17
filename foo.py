import numpy as np
import re

from config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--modelname", type=str, default="")
args = parser.parse_args()

def main():
  config = Config(args.dataset, args.modelname)
  print("data: %s, model: %s" % (config.dataset, config.modelname))

if __name__ == '__main__':
  main()


