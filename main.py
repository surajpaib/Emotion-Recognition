import torch
import logging

def main(args):





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", help="Path to train data")
    parser.add_argument("test_data", help="Path to test data")

    args = parser.parse_args()

    main(args)