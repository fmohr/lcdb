class IrreparableException(Exception):
    def __init__(self, cause, build_issue_text, objective_value, msg):
        super().__init__(msg)
        self.cause = cause
        self.build_issue_text = build_issue_text
        self.objective_value = objective_value


class AnticipatedMemoryError(IrreparableException):
    def __init__(self, msg):
        super().__init__(
            cause="anticipated_memory",
            build_issue_text="anticipated memory overflow",
            objective_value="F_anticipated_memory_error",
            msg=msg,
        )
