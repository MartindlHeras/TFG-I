package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
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

public class NetworkTrainer {
	
	private static final String DATASET_ROOT_FOLDER = "/home/martin/Documents/TFG_I/data/";
	private static final int N_SAMPLES_TRAINING = 30;
	private static final int N_SAMPLES_TESTING = 15;
	private static final int N_INPUTS = 6; // 9 si meto los que faltan
	private static final int N_OUTCOMES = 5;
	
	// CAPAS = 3
	private static final int N_HIDDEN = 50;
	private static final int nEpochs = 50;
	private static final int batchSize = 5;
	
	private static final double learningRate = 1e-4;
	private static final double lambda = 1e-7;
	
	private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
		
		INDArray input = Nd4j.create(new int[]{ nSamples, N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });
		
		int n = 0;
		try {
			Scanner myReader = new Scanner(new File(folderPath + "db.csv"));
			while (myReader.hasNextLine()) {
				String line = myReader.nextLine();
				String[] data = line.split(", ");
				input.putRow(n, Nd4j.createFromArray( new float[]{
						Integer.parseInt(data[1].substring(1)), // mutants
						Integer.parseInt(data[2].substring(1)), // tests
						Integer.parseInt(data[3].substring(1)), // cores
//						Integer.parseInt(data[4]), // tiempo total
						Integer.parseInt(data[5]), // tiempo original
//						Integer.parseInt(data[6]), // tiempo mutantes
//						Float.parseFloat(data[7]), // mutation score
						Integer.parseInt(data[8]), // lineas .c
						Integer.parseInt(data[9]), // size TS
						} ));
				output.putRow(n, crearSalida(Integer.parseInt(data[10].substring(1)), data[11]));
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
//		int batchSize = 1;
		//Build and return a data-set iterator that the network can use
		DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, batchSize );
		return dsi;
	}
	
	private static INDArray crearSalida(Integer alg, String opt) {
		int[] out = new int[N_OUTCOMES];
//		out[64*(alg-1)+Integer.parseInt(opt,2)] = 1;
		out[alg-1]=1;
		return Nd4j.createFromArray(out);
	}
	
	public static void main(String[] args) {
		long t0 = System.currentTimeMillis();
		DataSetIterator dsi = null;
		DataSetIterator testDsi = null;
		try {
			dsi = getDataSetIterator(DATASET_ROOT_FOLDER + "training/", N_SAMPLES_TRAINING);
		} catch (Exception e) { System.out.println(e); }
		try {
			testDsi = getDataSetIterator(DATASET_ROOT_FOLDER + "testing/", N_SAMPLES_TESTING);
		} catch (Exception e) { System.out.println(e); }
		
		int rngSeed = 123;
//		int nEpochs = 15; // Number of training epochs

		System.out.println("Build model....");
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		  .seed(rngSeed) //include a random seed for reproducibility
		  // use stochastic gradient descent as an optimization algorithm
		  .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		  .updater(new Adam(learningRate))
		  .l1(lambda)
		  .l2(lambda)
		  .list()
		  .layer(new DenseLayer.Builder() //create the first, input layer with Xavier initialization
				    .nIn(N_INPUTS)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new DenseLayer.Builder() //create the first, input layer with Xavier initialization
				    .nIn(N_HIDDEN)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new DenseLayer.Builder() //create the first, input layer with Xavier initialization
				    .nIn(N_HIDDEN)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new DenseLayer.Builder() //create the first, input layer with Xavier initialization
				    .nIn(N_HIDDEN)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new DenseLayer.Builder() //create the first, input layer with Xavier initialization
				    .nIn(N_HIDDEN)
				    .nOut(N_HIDDEN)
				    .activation(Activation.RELU)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
				    .nIn(N_HIDDEN)
				    .nOut(N_OUTCOMES)
				    .activation(Activation.SOFTMAX)
				    .weightInit(WeightInit.XAVIER)
				    .build())
		  .build();
		
//		EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder()
//		        .epochTerminationConditions(new MaxEpochsTerminationCondition(50))
//		        .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
//		        .scoreCalculator(new DataSetLossCalculator(testDsi, true))
//		        .evaluateEveryNEpochs(1)
//		        .modelSaver(new LocalFileModelSaver("/home/martin/Documents/TFG_I"))
//		        .build();
//
//		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,dsi);
//
//		//Conduct early stopping training:
//		EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
//
//		//Print out the results:
//		System.out.println("Termination reason: " + result.getTerminationReason());
//		System.out.println("Termination details: " + result.getTerminationDetails());
//		System.out.println("Total epochs: " + result.getTotalEpochs());
//		System.out.println("Best epoch number: " + result.getBestModelEpoch());
//		System.out.println("Score at best epoch: " + result.getBestModelScore());
//
//		//Get the best model:
//		MultiLayerNetwork bestModel = result.getBestModel();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		
		model.init();
		
		//print the score with every 250 iterations
		model.setListeners(new ScoreIterationListener(100));
		System.out.println("Train model....");
		model.fit(dsi, nEpochs);
		
		Evaluation eval = model.evaluate(dsi);
		System.out.println(eval.stats());
		
		System.out.println("Evaluate model....");
		eval = model.evaluate(testDsi);
		System.out.println(eval.stats());

		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		System.out.println("\n\nTotal time: "+t+" seconds");
		
		try {
			model.save(new File("model.dl4j"));
			System.out.printf("Model saved in model.dl4j");
		} catch (IOException e) {
			e.printStackTrace();
			System.out.printf("Error saving model");
		}
	}
}
