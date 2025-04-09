import json


def get_cli_test_command(entry):
    try:
        openmlid = int(entry['m:openmlid'])
    except:
        openmlid = None

    return (
        f"lcdb test"
        f" -i {openmlid}"
        f" -w {entry['m:workflow']}"
        f" --parameters='{json.dumps({k: v for k, v in entry.items() if k.startswith('p:')})}'"
    )