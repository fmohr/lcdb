import argparse


def parse_comma_separated_ints(value):
    try:
        # Split the input string by commas and convert each part to an integer
        values = [int(item.strip()) for item in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated list of integers: {value}") from e
    return values


def parse_comma_separated_strs(value):
    try:
        values = [item.strip() for item in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated list of strings: {value}") from e
    return values