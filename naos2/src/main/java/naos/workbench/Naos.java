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
	
//	private static final int N_INPUTS = 9;
	private static final int N_INPUTS = 6;
	
	private static List<String[]> fill(String fullFolderName, String appsFolder) {
		FileParser fp = null;
		List<String[]> inputs;
		List<String[]> fullInputs;
		System.out.println("Filling DBs...");
		try {
			fp = new FileParser(fullFolderName, appsFolder);
			inputs = fp.getInputs();
			fullInputs = fp.getFullInputs();
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e + " while handling the file");
			return null;
		}
		
		try {
			FileWriter fw = new FileWriter("db.txt", true);
			for (int i = 0; i < inputs.size(); i++) {
				fw.append(Arrays.toString(inputs.get(i)).substring(1, Arrays.toString(inputs.get(i)).length()-1) + "\n");
			}
			fw.close();
			System.out.println("DB filled!");
			fw = new FileWriter("fulldb.csv", true);
			fw.append("file name, mutants, tests, cores, total time, original time, mutants time, mutation score, TS size, lines, algorithm, optimizations\n");
			for (int i = 0; i < fullInputs.size(); i++) {
				fw.append(Arrays.toString(fullInputs.get(i)).substring(1, Arrays.toString(fullInputs.get(i)).length()-1) + "\n");
			}
			fw.close();
			System.out.println("Full DB filled!");
		} catch (IOException e) {
			System.out.println("Error while writing on db");
		}
		return inputs;
	}
	
	private static void train(String fullFolderName, String appsFolder) {
		List<String[]> inputs;
		
		inputs = fill(fullFolderName, appsFolder);
		
		System.out.println("Import model...");
		
		MultiLayerNetwork model = null;
		try {
			model = MultiLayerNetwork.load(new File("model.dl4j"), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("MultiLayerNetwork class not found");
			return;
		}
		
		// Esencialmente crear un dataset como en NetworkTraining

		INDArray input = Nd4j.create(new int[]{ inputs.size(), N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ inputs.size(), 320 });
		for (int i = 0; i < inputs.size(); i++) {
			
			input.putRow(0, Nd4j.createFromArray(new float[] {
				 									Integer.parseInt(inputs.get(i)[1].substring(1)), 
				 									Integer.parseInt(inputs.get(i)[2].substring(1)), 
				 									Integer.parseInt(inputs.get(i)[3].substring(1)),
//				 									Integer.parseInt(inputs.get(i)[4]),
				 									Integer.parseInt(inputs.get(i)[5]),
//				 									Integer.parseInt(inputs.get(i)[6]),
//				 									Float.parseFloat(inputs.get(i)[7]),
				 									Integer.parseInt(inputs.get(i)[8]),
				 									Integer.parseInt(inputs.get(i)[9])
				 									}));
			int[] label = new int[320];
			// indexOut = 64*(algoritmo - 1) + optimizaciones
			int indexOut = 64*(Integer.parseInt(inputs.get(i)[10].substring(1))-1)+Integer.parseInt(inputs.get(i)[11],2);
			label[indexOut] = 1;
			output.putRow(0, Nd4j.createFromArray(label));
		}
		DataSet dataSet = new DataSet( input, output );
		
		List<DataSet> listDataSet = dataSet.asList();
		DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, 1 );
		
		int nEpochs = 5;
		model.fit(dsi, nEpochs);
		
		return;
	}
	
	private static String predict(String fileName, String nMutants, String nTests, String nCores, String totalTime, String originalTime, String mutantsTime, String mutationScore, String TSSize, String lines) {
		
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
		INDArray output = model.output(createInput(nMutants,nTests,nCores,totalTime,originalTime,mutantsTime,mutationScore,TSSize,lines));
		
		float maxO = 0;
		int pos = 0;
		for (int i = 0; i < 320; i++) {
			if (output.getFloat(i) > maxO) {
				pos = i;
				maxO = output.getFloat(i);
			}
		}
		int alg = (pos/64)+1;
		int optInt = pos%64|64;
		
		return "a" + alg + ", " + Integer.toBinaryString(optInt).substring(1);
	}
	
	private static INDArray createInput(String nMutants, String nTests, String nCores, String totalTime, String originalTime, String mutantsTime, String mutationScore, String TSSize, String lines) {
		INDArray input = Nd4j.create(new int[]{ 1, N_INPUTS })
				 .putRow(0, Nd4j.createFromArray(new float[] {
			 									Integer.parseInt(nMutants), 
			 									Integer.parseInt(nTests), 
			 									Integer.parseInt(nCores),
//			 									Integer.parseInt(totalTime),
			 									Integer.parseInt(originalTime),
//			 									Integer.parseInt(mutantsTime),
//			 									Float.parseFloat(mutationScore),
			 									Integer.parseInt(TSSize),
			 									Integer.parseInt(lines)
			 									}));
		return input;
	}

	public static void main(String[] args) {
		if (args.length == 0) {
			System.out.println("Wrong command input, please select an option:");
			System.out.println("-t <fullFolderName> to train the Neural Network");
//			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> <totalTime> <originalTime> <mutantsTime> <mutationScore> <TSSize> <lines> to get the optimal execution mode");
			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> <originalTime> <TSSize> <lines> to get the optimal execution mode");
			return;
		}
		if (args[0].equals("-t")) {
			train(args[1], args[2]);
			System.out.println("ANN trained successfully");
		}
		else if (args[0].equals("-p")) {
			System.out.println("The best algorithm and optimizations for this situation are: " + predict(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]));
		}
		else if (args[0].equals("-f")) {
			fill(args[1], args[2]);
		}
		else {
			System.out.println("Wrong command input, please select an option:");
			System.out.println("-t <fullFolderName> <appsFolder> to train the Neural Network");
			System.out.println("-f <fullFolderName> <appsFolder> to fill the database");
//			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> <totalTime> <originalTime> <mutantsTime> <mutationScore> <TSSize> <lines> to get the optimal execution mode");
			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> <originalTime> <TSSize> <lines> to get the optimal execution mode");
		}
		return;
	}
}