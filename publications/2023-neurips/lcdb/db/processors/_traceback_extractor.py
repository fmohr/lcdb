import re
import json

from lcdb.analysis.views._util import get_cli_test_command


class TracebackExtractor:
    
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
        if "build_issues" in row and row["build_issues"] is not None:
            if isinstance(row["build_issues"], str):
                raise ValueError(f"Buid issues should already be unpacked, but got string: {row['build_issues']}")
            for anchor, traceback_at_anchor in row["build_issues"].items():
                errors.append({
                    "message": self.extract_error_message_from_traceback(traceback_at_anchor),
                    "location": f"anchor_{anchor}",
                    "traceback": traceback_at_anchor,
                    "cli_test_command": base_test_command + f" --anchor-schedule={anchor}"
                })
        
        # if there is a general traceback, also add it
        if "traceback" in row and isinstance(row["traceback"], str):
            errors.append({
                "message": self.extract_error_message_from_traceback(row["traceback"]),
                "location": f"global",
                "traceback": row["traceback"],
                "cli_test_command": base_test_command
            })
        return {"traceback_summary": errors if errors else None}


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