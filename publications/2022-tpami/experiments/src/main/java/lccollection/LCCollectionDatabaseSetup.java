package lccollection;


import java.io.File;

import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;

import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;

public class LCCollectionDatabaseSetup {

	public static void main(final String[] args) throws AlgorithmTimeoutedException, ExperimentDBInteractionFailedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		/* setup experimenter frontend */
		ExperimenterFrontend fe = new ExperimenterFrontend().withExperimentsConfig(new File("conf/experiments_lc_collection.conf")).withDatabaseConfig(new File("conf/database-local.conf"));
		fe.setLoggerName("frontend");
		fe.synchronizeDatabase();
	}
}
