class IrreparableException(Exception):
    """
    Exceptions of this type cannot be recovered at higher anchors
    """
    def __init__(self, cause, build_isssue_text, objective_value, msg):
        super().__init__(msg)
        self.cause = cause
        self.build_issue_text = build_isssue_text
        self.objective_value = objective_value

class AnticipatedMemoryError(IrreparableException):

    def __init__(self, msg):
        super().__init__(
            cause="anticipated_memory",
            build_isssue_text="anticipated memory overflow",
            objective_value="F_anticipated_memory_error",
            msg=msg
        )