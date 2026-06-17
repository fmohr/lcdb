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

        # check on timeout
        has_timeout = any([e["message"] == "timeout" for e in errors]) if errors else False
        anchor_with_timeout = int([e["location"][len("anchor_"):] for e in errors if e["message"] == "timeout"][0]) if has_timeout else None
        
        # summarize findings
        out = {"traceback_summary": errors if errors else None}
        out["timeout"] = has_timeout
        out["timeout_anchor"] = anchor_with_timeout
        out["anticipated memory overflow"] = any([e["message"] == "anticipated memory overflow" for e in errors]) if errors else False
        return out
