import json


def get_cli_test_command(entry):
    try:
        openmlid = int(entry['m:openmlid'])
    except:
        openmlid = None

    return (
        f"lcdb test"
        f" -i {openmlid}"
        f" -w {entry['workflow']}"
        f" --parameters='{json.dumps({k[2:]: v for k, v in entry.items() if k.startswith('p:')})}'"
        f" --log-level=debug"
        f" --no-exception-on-unsuitable-preprocessor"
        f" --suppress-json-output"
    )