package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
//import java.util.concurrent.TimeUnit;

//import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
//import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
//import org.deeplearning4j.earlystopping.EarlyStoppingResult;
//import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
//import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
//import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
//import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
//import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Network {
	
	private static final String DATASET_ROOT_FOLDER = "/home/martin/Documents/TFG_I/data/";
	private static final int N_INPUTS = 6;
	private static final int N_HIDDEN = 350;
	private static final int N_OUTCOMES = 320;
	
	private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
		
		INDArray input = Nd4j.create(new int[]{ nSamples, N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });
		
		int n = 0;
		try {
			Scanner myReader = new Scanner(new File(folderPath));
			while (myReader.hasNextLine()) {
				String line = myReader.nextLine();
				String[] data = line.split(", ");
				input.putRow(n, Nd4j.createFromArray( new float[]{
						Integer.parseInt(data[1].substring(1)), // mutants
						Integer.parseInt(data[2].substring(1)), // tests
						Integer.parseInt(data[3].substring(1)), // cores
						Integer.parseInt(data[5]), // tiempo original
						Integer.parseInt(data[10]), // lineas .c
						Integer.parseInt(data[11]), // size TS
						} ));
				output.putRow(n, crearSalida(Integer.parseInt(data[12].substring(1)), data[13]));
				n++;
			}
			myReader.close();
		} catch (FileNotFoundException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		//Join input and output matrixes into a data-set
		DataSet dataSet = new DataSet( input, output );
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(dataSet);
		normalizer.transform(dataSet);
		//Convert the data-set into a list
		List<DataSet> listDataSet = dataSet.asList();
		//Shuffle its content randomly
		Collections.shuffle( listDataSet, new Random(System.currentTimeMillis()) );
		//Set a batch size
		int batchSize = 10;
		//Build and return a data-set iterator that the network can use
		DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, batchSize );
		return dsi;
	}
	
	private static INDArray crearSalida(Integer alg, String opt) {
		int[] out = new int[N_OUTCOMES];
		out[64*(alg-1)+Integer.parseInt(opt,2)] = 1;
		return Nd4j.createFromArray(out);
	}
	
	public static void main(String[] args) {
		long t0 = System.currentTimeMillis();
		Double pct = 0.0;
		DataSetIterator dsi = null;
		DataSetIterator testDsi = null;
		String trainPath = DATASET_ROOT_FOLDER + (int)(pct*100) + "/augtraining" + (int)(pct*100) + ".csv";
		String testPath = DATASET_ROOT_FOLDER + (int)(pct*100) + "/augtesting" + (int)(pct*100) + ".csv";
		
		try {
			dsi = getDataSetIterator(trainPath, (int) Files.lines(Paths.get(trainPath)).count());
			testDsi = getDataSetIterator(testPath, (int) Files.lines(Paths.get(testPath)).count());
		} catch (Exception e) { System.out.println(e); }
		try {
		} catch (Exception e) { System.out.println(e); }
		
		int rngSeed = 123;
		int nEpochs = 50;

		System.out.println("Build model....");
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		  .seed(rngSeed)
		  .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		  .updater(new Adam(1e-3))
		  .l1(1e-6)
		  .l2(1e-6)
		  .list()
		  .layer(new DenseLayer.Builder()
				    .nIn(N_INPUTS)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new DenseLayer.Builder()
				    .nIn(N_HIDDEN)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new DenseLayer.Builder()
				    .nIn(N_HIDDEN)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
				    .nIn(N_HIDDEN)
				    .nOut(N_OUTCOMES)
				    .activation(Activation.SOFTMAX)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		
		model.init();
		
		//print the score with every 1000 iterations
		model.setListeners(new ScoreIterationListener(1000));
		System.out.println("Train model...");
		model.fit(dsi, nEpochs);
		
		Evaluation eval = model.evaluate(dsi);
		System.out.println(eval.stats());
		
		System.out.println("Evaluate model...");
		eval = model.evaluate(testDsi);
		System.out.println(eval.stats());
		
		try {
			model.save(new File("model" + (int)(pct*100) + "aug.dl4j"));
			System.out.printf("Model saved in model" + (int)(pct*100) + "aug.dl4j\n");
		} catch (IOException e) {
			e.printStackTrace();
			System.out.printf("Error saving model");
		}
		
		System.out.println("Accuracy: " + Tester.testNetwork(pct, pct, testPath));
		
		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		System.out.println("\n\nTotal time: "+t+" seconds");
	}
}
