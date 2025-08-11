import re
import json

from lcdb.analysis.views._util import get_cli_test_command


class TracebackExtractor:

    def __init__(self):
        self.rows = []
    
    def extract_error_message_from_traceback(self, traceback_str):

        # extract errors from traceback messages str format first
        try:
            error_message = re.search(r'(\w+Error): (.*)', traceback_str).group(0)
        except:
            error_message = traceback_str
        return error_message

    def __call__(self, row):
        """
            Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.
        """
        base_test_command = get_cli_test_command(row)
        errors = []

        # if there are build issues, add them
        if "m:build_issues" in row and isinstance(row["m:build_issues"], str):
            for anchor, traceback_at_anchor in json.loads(row["m:build_issues"]).items():
                errors.append({
                    "message": self.extract_error_message_from_traceback(traceback_at_anchor),
                    "location": f"anchor_{anchor}",
                    "traceback": traceback_at_anchor,
                    "cli_test_command": base_test_command + f" --anchor-schedule={anchor}"
                })
        
        # if there is a general traceback, also add it
        if "m:traceback" in row and isinstance(row["m:traceback"], str):
            errors.append({
                "message": self.extract_error_message_from_traceback(row["m:traceback"]),
                "location": f"global",
                "traceback": row["m:traceback"],
                "cli_test_command": base_test_command
            })
        return errors if errors else None


if __name__ == "__main__":

    from lcdb.db import LCDB

    lcdb = LCDB()
    df = lcdb.query(
        workflows=["lcdb.workflow.sklearn.LibLinearWorkflow"],
        openmlids=[3, 1111],
        test_seeds=[0],
        return_generator=False,
        processors={
            "tracebacks": TracebackExtractor()
        },
        show_progress=True
    )
    print(df)