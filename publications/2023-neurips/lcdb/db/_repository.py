from abc import abstractmethod, ABC


class Repository(ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get(path):
        from ._local_repository import LocalRepository
        from ._pcloud_repository import PCloudRepository

        if path.startswith("pcloud://"):
            repo_code = path[9:]
            return PCloudRepository(repo_code=repo_code)

        else:
            return LocalRepository(path)

    @abstractmethod
    def add_results(self, campaign, *result_files):
        raise NotImplementedError

    @abstractmethod
    def get_workflows(self):
        raise NotImplementedError

    @abstractmethod
    def get_campaigns(self, workflow):
        raise NotImplementedError

    @abstractmethod
    def get_datasets(self, workflow, campaign):
        raise NotImplementedError

    @abstractmethod
    def get_count_table(self):
        """

        A pandas dataframe with the number of result rows for each combinations of workflow, dataset, campaign, test seed, validation seed, workflow seed

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def query_results_as_stream(
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
        :return: Pandas dataframe or generator thereof with all results observed that match *all* the given criteria
        """
        raise NotImplementedError
