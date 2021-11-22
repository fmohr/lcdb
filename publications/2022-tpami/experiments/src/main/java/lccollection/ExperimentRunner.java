package lccollection;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.processes.ProcessIDNotRetrievableException;
import ai.libs.jaicore.processes.ProcessUtil;

public class ExperimentRunner implements IExperimentSetEvaluator {

	private static final Logger logger = LoggerFactory.getLogger("experimenter");

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		try {

			/* get configuration */
			Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
			int openmlid = Integer.valueOf(keys.get("openmlid"));
			int outer_seed= Integer.valueOf(keys.get("outer_seed"));
			int inner_seed_index = Integer.valueOf(keys.get("inner_seed_index"));
			String algo = keys.get("learner");
			logger.info("\topenmlid: {}", openmlid);
			logger.info("\tlearner: {}", algo);
			logger.info("\touter seed: {}", outer_seed);
			logger.info("\tinner seed index: {}", inner_seed_index);

			/* run python experiment */
			String options = openmlid + " " + algo + " " + outer_seed + " " + inner_seed_index;

			/* write results */
			Map<String, Object> map = new HashMap<>();
			map.put("result", getPythonExperimentResults(options));
			processor.processResults(map);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public static String getPythonExperimentResults(final String options) throws InterruptedException, IOException, ProcessIDNotRetrievableException {
		File workingDirectory = new File("python");
		String id = UUID.randomUUID().toString();
		File outFile = new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id + ".json");
		outFile.getParentFile().mkdirs();

		File file = new File("computelc.py");
		String singularityImage = "test.simg";
		List<String> args = Arrays.asList("singularity", "exec", singularityImage, "bash", "-c", "python3 " + file + " " + options + " " + outFile.getAbsolutePath());
		logger.info("Executing {} in singularity.", new File(workingDirectory + File.separator + file));
		logger.info("Arguments: {}", args);

		ProcessBuilder pb = new ProcessBuilder(args);
		pb.directory(workingDirectory);
		pb.redirectErrorStream(true);
		System.out.println("Starting process. Current memory usage is " + ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024) + "MB");
		Process p = pb.start();
		System.out.println("PID: " + ProcessUtil.getPID(p));
		try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
			String line;
			while ((line = br.readLine()) != null) {
				System.out.println(" --> " + line);
			}

			System.out.println("awaiting termination");
			while (p.isAlive()) {
				Thread.sleep(1000);
			}
			System.out.println("ready");

			return FileUtil.readFileAsString(outFile);
		}
		finally {
			System.out.println("KILLING PROCESS!");
			ProcessUtil.killProcess(p);
		}
	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, InterruptedException, ExperimentEvaluationFailedException, ExperimentFailurePredictionException {

		String databaseconf = args[0];
		String jobInfo = args[1];

		/* setup experimenter frontend */
		ExperimenterFrontend fe = new ExperimenterFrontend().withEvaluator(new ExperimentRunner()).withExperimentsConfig(new File("conf/experiments_lc_collection.conf")).withDatabaseConfig(new File(databaseconf));
		fe.setLoggerName("frontend");
		fe.withExecutorInfo(jobInfo);


		long deadline = System.currentTimeMillis() + 86000 * 1000;
		long remainingTime;
		do {
			logger.info("Conducting experiment. Currently used memory is {}MB. Free memory is {}MB.", (Runtime.getRuntime().maxMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024.0), Runtime.getRuntime().freeMemory() / (1024 * 1024.0));
			fe.randomlyConductExperiments(1);
			remainingTime = deadline - System.currentTimeMillis();
			logger.info("Experiment finished. Remaining time: {}!", remainingTime);
		}
		while (remainingTime > 12 * 3600 * 1000);
		System.out.println("Remaining time only " + (remainingTime / 1000) + "s. Stopping.");
	}
}
