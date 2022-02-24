package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class Naos {
	
	private static void train(String fullFolderName) {
		FileParser fp = null;
		String[] inputs;
		
		try {
			fp = new FileParser(fullFolderName);
			inputs = fp.getInputs();
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e + " while handling the file");
			return;
		}
		
		try {
			FileWriter fw = new FileWriter("db.txt", true);
			fw.append(Arrays.toString(inputs).substring(1, Arrays.toString(inputs).length()-1) + "\n");
			fw.close();
		} catch (IOException e) {
			System.out.println("Error while writing on db");
		}
		
		System.out.println("Import model....");
		
		MultiLayerNetwork model = null;
		try {
			model = MultiLayerNetwork.load(new File("model.dl4j"), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("MultiLayerNetwork class not found");
			return;
		}
		
		// Esencialmente crear un dataset como en NetworkTraining pero solo con un elemento
		INDArray input = createInput(inputs[1].substring(1),inputs[2].substring(1),inputs[3].substring(1));
		int[] label = new int[5];
		label[Integer.parseInt(inputs[4].substring(1))-1] = 1;
		INDArray output = Nd4j.create(new int[]{ 1, 5 }).putRow(0, Nd4j.createFromArray(label));
		DataSet dataSet = new DataSet( input, output );
		
		List<DataSet> listDataSet = dataSet.asList();
		DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, 1 );
		
		int nEpochs = 5;
		model.fit(dsi, nEpochs);
		
		return;
	}
	
	private static String predict(String fileName, String nMutants, String nTests, String nCores) {
		
		System.out.println("Import model....");
		
		MultiLayerNetwork model = null;
		try {
			model = MultiLayerNetwork.load(new File("model.dl4j"), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("MultiLayerNetwork class not found");
			return null;
		}
		
		// Obtener e interpretar el output a traves del modelo que ya tenemos
		INDArray output = model.output(createInput(nMutants,nTests,nCores));
		float maxO = 0;
		int alg = 0;
		for (int i = 0; i < 5; i++) {
			if (output.getFloat(i) > maxO) {
				alg = i+1;
				maxO = output.getFloat(i);
			}
		}
		
		return "a" + alg;
	}
	
	private static INDArray createInput(String nMutants, String nTests, String nCores) {
		INDArray input = Nd4j.create(new int[]{ 1, 3 })
				 .putRow(0, Nd4j.createFromArray(new int[] {
			 									Integer.parseInt(nMutants), 
			 									Integer.parseInt(nTests), 
			 									Integer.parseInt(nCores)
			 									}));
		return input;
	}

	public static void main(String[] args) {
		if (args.length == 0) {
			System.out.println("Wrong command input, please select an option:");
			System.out.println("-t <fullFolderName> to train the Neural Network");
			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> to get the optimal execution mode");
			return;
		}
		if (args[0].equals("-t")) {
			train(args[1]);
			System.out.println("ANN trained successfully");
		}
		else if (args[0].equals("-p")) {
			System.out.println("The optimal algorithm for this situation is: " + predict(args[1], args[2], args[3], args[4]));
		}
		else {
			System.out.println("Wrong command input, please select an option:");
			System.out.println("-t <fullFolderName> to train the Neural Network"); // Tengo que transformar para que funcione tambien con paths locales
			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> to get the optimal execution mode"); // Buscar un modo con menos inputs
		}
		return;
	}
}
