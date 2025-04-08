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


  
def get_true_mean(df, workflow, openmlid, column_name):
    # filter dataframe by workflow and openmlid
    filtered_df = df[(df["workflow"] == workflow) & (df["openmlid"] == openmlid)]

    if not filtered_df.empty:
        return filtered_df[f"mean_{column_name}"].values[0]
    else:
        return None


def get_cores(memory_per_core, number_of_nodes):
    node_cores = 192
    node_mem = 336
    total_memory = node_mem * number_of_nodes
    # calculate the memory per core
    total_cores = node_cores * number_of_nodes
    # calculate the number of cores
    cores = int(min(total_memory // memory_per_core, total_cores))
    return cores