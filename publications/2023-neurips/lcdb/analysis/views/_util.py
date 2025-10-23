import json


def get_cli_test_command(entry):
    openmlid = int(entry['openmlid'])
    return (
        f"lcdb test"
        f" -i {openmlid}"
        f" -w {entry['workflow']}"
        f" --parameters='{json.dumps(entry['config'])}'"
        f" --log-level=debug"
        f" --no-exception-on-unsuitable-preprocessor"
        f" --suppress-json-output"
    )