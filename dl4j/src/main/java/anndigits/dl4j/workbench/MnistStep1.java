package anndigits.dl4j.workbench;
import java.io.File;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.datavec.image.loader.NativeImageLoader;
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
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

public class MnistStep1 implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -884560880506197520L;
	//The absolute path of the folder containing MNIST training and testing sub-folders
	private static final String MNIST_DATASET_ROOT_FOLDER = "/home/martin/Documents/mnist_png/";
	//Height and width in pixel of each image
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	//The total number of images into the training and testing set
	private static final int N_SAMPLES_TRAINING = 60000;
	private static final int N_SAMPLES_TESTING = 10000;
	//The number of possible outcomes of the network for each input, 
	//correspondent to the 0..9 digit classification
	private static final int N_OUTCOMES = 10;
	
	private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
		File folder = new File(folderPath);
		File[] digitFolders = folder.listFiles();
		NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH);
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);

		INDArray input = Nd4j.create(new int[]{ nSamples, HEIGHT*WIDTH });
		INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });
		
		int n = 0;
		//scan all 0..9 digit sub-folders
		for (File digitFolder : digitFolders) {
		  //take note of the digit in processing, since it will be used as a label
		  int labelDigit = Integer.parseInt(digitFolder.getName());
		  //scan all the images of the digit in processing
		  File[] imageFiles = digitFolder.listFiles();
		  for (File imageFile : imageFiles) {
		    //read the image as a one dimensional array of 0..255 values
		    INDArray img = nil.asRowVector(imageFile);
		    //scale the 0..255 integer values into a 0..1 floating range
		    //Note that the transform() method returns void, since it updates its input array
		    scaler.transform(img);
		    //copy the img array into the input matrix, in the next row
		    input.putRow( n, img );
		    //in the same row of the output matrix, fire (set to 1 value) the column correspondent to the label
		    output.put( n, labelDigit, 1.0 );
		    //row counter increment
		    n++;
		  }
		}
		
		//Join input and output matrixes into a data-set
		DataSet dataSet = new DataSet( input, output );
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
	
	public static void main(String[] args) {
		long t0 = System.currentTimeMillis();
		DataSetIterator dsi = null;
		DataSetIterator testDsi = null;
		try {
			dsi = getDataSetIterator(MNIST_DATASET_ROOT_FOLDER + "training", N_SAMPLES_TRAINING);
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
		  .layer(new DenseLayer.Builder() //create the first, input layer with xavier initialization
		    .nIn(HEIGHT*WIDTH)
		    .nOut(1000)
		    .activation(Activation.RELU)
		    .weightInit(WeightInit.XAVIER)
		    .build())
		  .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
		    .nIn(1000)
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
			testDsi = getDataSetIterator( MNIST_DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);
		} catch (Exception e) { System.out.println(e); }
		
		System.out.println("Evaluate model....");
		Evaluation eval = model.evaluate(testDsi);
		System.out.println(eval.stats());

		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		System.out.println("\n\nTotal time: "+t+" seconds");
		
		try {
	         FileOutputStream fileOut =
	         new FileOutputStream("/home/martin/Documents/TFG_I/dl4j/model.ser");
	         ObjectOutputStream out = new ObjectOutputStream(fileOut);
	         out.writeObject(model);
	         out.close();
	         fileOut.close();
	         System.out.printf("Serialized data is saved in /home/martin/Documents/TFG_I/dl4j/model.ser");
	      } catch (IOException i) { i.printStackTrace(); }
	}
}
