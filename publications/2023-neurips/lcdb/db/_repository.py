import os
from abc import abstractmethod, ABC


class Repository(ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get(path):
        from ._local_repository import LocalRepository
        return LocalRepository(path)

    @abstractmethod
    def add_results(self, campaign, *result_files):
        raise NotImplementedError

    @abstractmethod
    def get_workflows(self, campaign=None):
        raise NotImplementedError

    @abstractmethod
    def get_datasets(self, campaign, workflow):
        raise NotImplementedError

    @abstractmethod
    def get_result_files_of_workflow_and_dataset_in_campaign(
            self,
            campaign,
            workflow,
            openmlid,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        raise NotImplementedError

    @abstractmethod
    def get_result_files_of_workflow_in_campaign(
            self,
            campaign,
            workflow,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        raise NotImplementedError

    @abstractmethod
    def get_result_files_of_workflow(
            self,
            workflow=None,
            campaigns=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        raise NotImplementedError

    @abstractmethod
    def get_results(
            self,
            campaigns=None,
            workflows=None,
            openmlids=None,
            workflow_seeds=None,
            test_seeds=None,
            validation_seeds=None
    ):
        """

        :param campaigns:
        :param workflows:
        :param openmlids:
        :param workflow_seeds:
        :param test_seeds:
        :param validation_seeds:
        :return: dataframe with all results observed that match *all* the given criteria
        """
        raise NotImplementedError
