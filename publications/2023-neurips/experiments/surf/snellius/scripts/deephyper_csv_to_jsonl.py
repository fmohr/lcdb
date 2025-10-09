from lcdb.builder.utils import deephyper_results_to_jsonl

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:
        print("Usage: python deephyper_csv_to_jsonl.py <input_csv> <output_jsonl>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_jsonl = sys.argv[2]

    deephyper_results_to_jsonl(input_csv, output_jsonl)