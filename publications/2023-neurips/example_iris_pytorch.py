from lcdb.workflow.torch import SimpleTorchMLPWorkflow
from lcdb.workflow._runner import WorkflowRunner

results = WorkflowRunner.run(
    openmlid=61, #iris
    workflow_class=SimpleTorchMLPWorkflow,
    anchor=100,
    monotonic=False,
    inner_seed=0,
    outer_seed=0,
    hyperparameters={}
)

# Collect all data that should be JSON Serializable
print(results)
