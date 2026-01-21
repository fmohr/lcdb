import re

def extract_anticipated_memory(row):
    """
    Extract anticipated memory values (predicted and max in GB) from m:json if present.
    Returns just the predicted GB if found, otherwise None.
    """
    
    # parse the JSON (it's stored as string in row["m:json"])
    data = row["results"]
    if data is None:
        return {"anticipated_memory": None}
    if not isinstance(data, dict):
        raise ValueError(f"results must be unpacked to use memory anticipation.")

    # recursive search through dicts/lists for traceback
    def find_traceback(obj):
        if isinstance(obj, dict):
            if "traceback" in obj.get("metadata", {}):
                return obj["metadata"]["traceback"]
            for child in obj.get("children", []):
                res = find_traceback(child)
                if res:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = find_traceback(item)
                if res:
                    return res
        return None

    traceback = find_traceback(data)
    if traceback:
        # regex to capture GB values
        match = re.search(
            r"consume approximately ([0-9.]+) GB.*?maximum is ([0-9.]+) GB",
            traceback,
            re.DOTALL,
        )
        if match:
            predicted, maximum = match.groups()
            return {"anticipated_memory": float(predicted)}
    return {"anticipated_memory": None}