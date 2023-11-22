import os
import sys

if __name__ == "__main__":
    ranks_per_node = int(os.environ.get("NRANKS_PER_NODE", 4))
    fname = os.environ["PBS_NODEFILE"]
    output = ""
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            line = line.strip("\n")
            if i == 0:
                output += f"{line}"
            for _ in range(ranks_per_node):
                output += f",{line}"
    print(output)
