package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class NetworkTrainer {
	
	private static final String DATASET_ROOT_FOLDER = "/home/martin/Documents/TFG_I/naos/";
	private static final int N_SAMPLES_TRAINING = 9;
	private static final int N_SAMPLES_TESTING = 9;
	private static final int N_INPUTS = 7;
	private static final int N_OUTCOMES = 320;
	
	// IMPORTANTE ADAPTAR AL PATH DE TRAINING Y TESTS QUE NO HE HECHO
	private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {

		INDArray input = Nd4j.create(new int[]{ nSamples, N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });
		
		int n = 0;
		try {
			Scanner myReader = new Scanner(new File("db.txt"));
			while (myReader.hasNextLine()) {
				String line = myReader.nextLine();
				String[] data = line.split(", ");
				input.putRow(n, Nd4j.createFromArray( new float[]{
						Integer.parseInt(data[1].substring(1)), // mutants
						Integer.parseInt(data[2].substring(1)), // tests
						Integer.parseInt(data[3].substring(1)), // cores
						Integer.parseInt(data[4]), // tiempo total
						Integer.parseInt(data[5]), // tiempo original
						Integer.parseInt(data[6]), // tiempo mutantes
						Float.parseFloat(data[7]), // mutation score
						} ));
				output.putRow(n, crearSalida(Integer.parseInt(data[8].substring(1)), data[9]));
				n++;
				System.out.println("input: " + input);
				System.out.println("output: " + output);
			}
			myReader.close();
	    } catch (FileNotFoundException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
	    }
		// No estoy seguro de si normalizar los datos antes de meterlos, problematico a la hora de entrenar solo con un modelo
//		int maxMutants = Collections.max(mutants);
//		int maxTests = Collections.max(tests);
//		int maxCores = Collections.max(cores);
		
		//Join input and output matrixes into a data-set
		DataSet dataSet = new DataSet( input, output );
		//Convert the data-set into a list
		List<DataSet> listDataSet = dataSet.asList();
		//Shuffle its content randomly
		Collections.shuffle( listDataSet, new Random(System.currentTimeMillis()) );
		//Set a batch size
		int batchSize = 1;
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
		DataSetIterator dsi = null;
		DataSetIterator testDsi = null;
		try {
			dsi = getDataSetIterator(DATASET_ROOT_FOLDER + "training", N_SAMPLES_TRAINING);
		} catch (Exception e) { System.out.println(e); }
		
		
		int rngSeed = 123;
		int nEpochs = 2; // Number of training epochs

		System.out.println("Build model....");
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		  .seed(rngSeed) //include a random seed for reproducibility
		  // use stochastic gradient descent as an optimization algorithm
		  .updater(new Nesterovs(0.006, 0.9))
		  .l2(1e-4)
		  .list()
		  .layer(new DenseLayer.Builder() //create the first, input layer with Xavier initialization
		    .nIn(N_INPUTS)
		    .nOut(325)
		    .activation(Activation.RELU)
		    .weightInit(WeightInit.XAVIER)
		    .build())
		  .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
		    .nIn(325)
		    .nOut(N_OUTCOMES)
		    .activation(Activation.SOFTMAX)
		    .weightInit(WeightInit.XAVIER)
		    .build())
		  .build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		//print the score with every 500 iteration
		model.setListeners(new ScoreIterationListener(500));
		System.out.println("Train model....");
		model.fit(dsi, nEpochs);
		
		try {
			testDsi = getDataSetIterator(DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);
		} catch (Exception e) { System.out.println(e); }
		
		System.out.println("Evaluate model....");
		Evaluation eval = model.evaluate(testDsi);
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
