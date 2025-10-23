import importlib
import numpy as np

def get_workflow_class(workflow_class_as_str: str):
    """Import an attribute from a module given its path.

    Example:

    >>> from lcdb.workflow import SVMWorkflow
    >>> import_attr_from_module("lcdb.workflow.SVMWorkflow") == SVMWorkflow
    """
    if not workflow_class_as_str.startswith("lcdb.workflow"):
        raise ValueError(f"Can only import a module that starts with lcdb.workflow")
    workflow_class_as_str = workflow_class_as_str.split(".")
    module_name, attr_name = (
        ".".join(workflow_class_as_str[:-1]),
        workflow_class_as_str[-1],
    )
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    return attr

def get_config_space_of_workflow(workflow_class_as_str: str):
    return get_workflow_class(workflow_class_as_str).config_space()

def get_default_config(config_space):
    defaulthps = {}
    for (
        hyperparameter_name,
        hyperparameter,
    ) in config_space.get_hyperparameters_dict().items():
        default = hyperparameter.default_value
        defaulthps[hyperparameter_name] = default
    default_config = ConfigSpace.configuration_space.Configuration(
        config_space, values=defaulthps, allow_inactive_with_values=True
    )
    # default_config = config_space.deactivate_inactive_hyperparameters(default_config, config_space) # only available in later version...
    return default_config
