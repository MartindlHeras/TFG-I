package anndigits.dl4j.workbench;
import java.io.File;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

public class TestANN {
	
	//The absolute path of the folder containing MNIST training and testing sub-folders
	private static final String MNIST_DATASET_ROOT_FOLDER = "/home/martin/Documents/mnist_png/";
	//Height and width in pixel of each image
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	//The total number of images into the training and testing set
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
		DataSetIterator testDsi = null;

		System.out.println("Import model....");
		
		MultiLayerNetwork model = null;
		try {
		   FileInputStream fileIn = new FileInputStream("/home/martin/Documents/TFG_I/dl4j/model.ser");
		   ObjectInputStream in = new ObjectInputStream(fileIn);
		   model = (MultiLayerNetwork) in.readObject();
		   in.close();
		   fileIn.close();
		} catch (IOException i) {
		   i.printStackTrace();
		   return;
		} catch (ClassNotFoundException c) {
		   System.out.println("MultiLayerNetwork class not found");
		   c.printStackTrace();
		   return;
		}
		
		try {
			testDsi = getDataSetIterator( MNIST_DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);
		} catch (Exception e) { System.out.println(e); }
		
		System.out.println("Evaluate model....");
		Evaluation eval = model.evaluate(testDsi);
		System.out.println(eval.stats());

		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		System.out.println("\n\nTotal time: "+t+" seconds");
	}
}

