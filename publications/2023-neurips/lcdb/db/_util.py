import os.path
import pathlib


def get_path_to_lcdb(lcdb_folder_name=".lcdb"):

    # by default we look if `.lcdb_config.json` exist in the current directory
    if pathlib.Path(lcdb_folder_name).exists():
        return f"{os.getcwd()}/{lcdb_folder_name}"

    # If it does not exist we fall back to ~/.lcdb
    default_path = pathlib.Path(os.path.expanduser(f"~/{lcdb_folder_name}"))
    return default_path.absolute()


class CountAwareGenerator:
    def __init__(self, n, gen):
        self.n = n  # Store the total number of elements
        self.gen = gen

    def __iter__(self):
        for i in self.gen:
            yield i

    def __len__(self):
        return self.n  # Return the total number of elements


def print_tree(node, indent="", last='updown'):
    """Recursively prints a tree from a dictionary in a tree-like structure."""
    if isinstance(node, dict):
        children = node["children"] if "children" in node else []
        for i, child in enumerate(children):
            updown = 'up' if i == 0 else ('down' if i == len(children) - 1 else 'updown')
            connector = "└── " if updown == 'down' else "├── "
            print(indent + connector + str(child["tag"]) if "tag" in child else "<unknown>")
            if "children" in child:
                extension = "    " if updown == 'down' else "│   "
                print_tree(child, indent + extension, updown)