
def compute_payload(row):
    return {"payload": len(str(row["results"])) if row["results"] is not None else 0}

