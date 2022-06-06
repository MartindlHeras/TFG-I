package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class Trainer {
	
	private static final int TOTAL_INPUTS = 12;
	private static final int N_INPUTS = 6; // 9 si meto los que faltan
	private static final String localPath = "/home/martin/Documents/TFG_I";
	
	private static List<String[]> getInputs(String dataPath) throws FileNotFoundException {
		List<String[]> inputs = new ArrayList<String[]>();
		File dataDirectory = new File(dataPath);
		// input -> [nombre,mutantes,tests,cores,tiempo,tiempo original,tiempo mutantes,mutation score,lineas c,size TS,algoritmo,optimizaciones]
		
		if (!dataDirectory.exists()) {
			return null;
		}
		
		for (final File fileName : dataDirectory.listFiles()) {
			String[] input = new String[TOTAL_INPUTS];
			double minTime = Double.MAX_VALUE;
			double maxMS = 0;
			String[] parts = fileName.getName().split("_");
			boolean flag = false;
			// Asignamos nombre del programa, #mutantes, #tests, #cores, algoritmo inicial y optimizaciones iniciales
			for (int i = 0; i < 4; i++) {
			      input[i] = parts[5+i];
			}
			input[10] = parts[9]; // Algoritmo inicial
			input[11] = parts[10]; // Optimizaciones
			
			// Comprobamos si hemos pasado ya por ese caso de app mutantes tests y cores, si hemos pasado, pasamos al siguiente
			for (int i = 0; i < inputs.size(); i++) {
				if (input[0].equals(inputs.get(i)[0]) && input[1].equals(inputs.get(i)[1]) && input[2].equals(inputs.get(i)[2]) && input[3].equals(inputs.get(i)[3])) { 
					// Si ya he pasado por esas especificaciones
					flag = true;
					break;
				}
			}
			if (flag) {
				continue;
			}
			// Recorre todos los ficheros que coinciden en programa, mutantes, tests y cores
			for (final File fileEntry : dataDirectory.listFiles()) {
				String[] tmp = fileEntry.getName().split("_");
				
				if (input[0].equals(tmp[5]) && input[1].equals(tmp[6]) && input[2].equals(tmp[7]) && input[3].equals(tmp[8])) {
					// Si el tiempo del fichero es menor que el que tenemos, actualiza el minimo, algoritmo y optimizaciones
					File f = new File(dataPath + "/" + fileEntry.getName() + "/malone_overview.txt");
					Scanner scan = new Scanner(f);
					double totalTime = 0;
					String originalTime = null, mutantsTime = null, mutationScore = null;
					while (scan.hasNextLine()) {
						String[] tmpFileTime = scan.nextLine().split(":");
						// Recorremos hasta llegar a la linea que nos interesa y devolvemos el valor del tiempo
						if (tmpFileTime[0].equals("Total time")) { // RegEx
							totalTime = Double.parseDouble(tmpFileTime[1]);
						}
						if (tmpFileTime[0].equals("Original time")) { // RegEx
							originalTime = tmpFileTime[1];
						}
						if (tmpFileTime[0].equals("Mutants time")) { // RegEx
							mutantsTime = tmpFileTime[1];
						}
						if (tmpFileTime[0].equals("Mutation score")) { // RegEx
							mutationScore = tmpFileTime[1];
						}
					}
					scan.close();
					if (maxMS < Double.parseDouble(mutationScore.substring(1)) &&  minTime > totalTime) {
						maxMS = Double.parseDouble(mutationScore.substring(1));
						minTime = totalTime;
						input[4] = Integer.toString((int)totalTime);
						input[5] = originalTime.substring(1);
						input[6] = mutantsTime.substring(1);
						input[7] = mutationScore.substring(1);
						input[10] = tmp[9];
						input[11] = tmp[10];
					}
				}
			}
			// appsFolder = /home/martin/Documents/TFG_I/apps
			Path cPath = Paths.get(localPath + "/apps/" + input[0] + "/" + input[0]+ ".c");
			Path TSPath = Paths.get(localPath + "/apps/" + input[0] + "/tests_" + input[0] + ".txt");
			try {
				input[8] = Long.toString(Files.size(TSPath)); // TAMANO TS
				input[9] = Long.toString(Files.lines(cPath).count()); // NUMERO DE LINEAS
			} catch (IOException e) {
				e.printStackTrace();
			}
			inputs.add(input);
		}
		return inputs;
	}
	
	private static List<String[]> getFullInputs(String dataPath) throws FileNotFoundException {
		List<String[]> inputs = new ArrayList<String[]>();
		File dataDirectory = new File(dataPath);
		// input -> [nombre,mutantes,tests,cores,tiempo,tiempo original,tiempo mutantes,mutation score,lineas c,size TS,algoritmo,optimizaciones]
		
		if (!dataDirectory.exists()) {
			return null;
		}
		
		for (final File fileName : dataDirectory.listFiles()) {
			String[] input = new String[TOTAL_INPUTS];
			String[] parts = fileName.getName().split("_");
			// Asignamos nombre del programa, #mutantes, #tests, #cores, algoritmo inicial y optimizaciones iniciales
			for (int i = 0; i < 4; i++) {
			      input[i] = parts[5+i];
			}
			input[10] = parts[9]; // Algoritmo inicial
			input[11] = parts[10]; // Optimizaciones
			
			File f = new File(dataPath + "/" + fileName.getName() + "/malone_overview.txt");
			Scanner scan = new Scanner(f);
			while (scan.hasNextLine()) {
				String[] tmpFileTime = scan.nextLine().split(":");
				// Recorremos hasta llegar a la linea que nos interesa y devolvemos el valor del tiempo
				if (tmpFileTime[0].equals("Total time")) {
					input[4] = tmpFileTime[1].substring(1);
				}
				if (tmpFileTime[0].equals("Original time")) {
					input[5] = tmpFileTime[1].substring(1);
				}
				if (tmpFileTime[0].equals("Mutants time")) {
					input[6] = tmpFileTime[1].substring(1);
				}
				if (tmpFileTime[0].equals("Mutation score")) {
					input[7] = tmpFileTime[1].substring(1);
				}
			}
			scan.close();			
			
			// appsFolder = /home/martin/Documents/TFG_I/apps
			Path cPath = Paths.get(localPath + "/apps/" + input[0] + "/" + input[0]+ ".c");
			Path TSPath = Paths.get(localPath + "/apps/" + input[0] + "/tests_" + input[0] + ".txt");
			try {
				input[8] = Long.toString(Files.size(TSPath)); // TAMANYO TS
				input[9] = Long.toString(Files.lines(cPath).count()); // NUMERO DE LINEAS
			} catch (IOException e) {
				e.printStackTrace();
			}
			inputs.add(input);
		}
		return inputs;
	}

	protected static List<String[]> fill(String dataPath) {
		
		List<String[]> inputs;
		List<String[]> fullInputs;
		// System.out.println("Filling DBs...");
		
		if (dataPath == null) {
			return null;
		}
		
		try {
			inputs = getInputs(dataPath);
			fullInputs = getFullInputs(dataPath);
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e + " while handling the file");
			return null;
		}
		
		if (inputs == null || fullInputs == null) {
			return null;
		}
		
		try {
			FileWriter fw = new FileWriter(localPath + "/data/training/db.csv", true);
			for (int i = 0; i < inputs.size(); i++) {
				fw.append(Arrays.toString(inputs.get(i)).substring(1, Arrays.toString(inputs.get(i)).length()-1) + "\n");
			}
			fw.close();
			// System.out.println("DB filled!");
			fw = new FileWriter(localPath + "/data/fulldb.csv", true);
			if ((new File(localPath + "/data/fulldb.csv")).length() == 0) {
				fw.append("file name, mutants, tests, cores, total time, original time, mutants time, mutation score, TS size, lines, algorithm, optimizations\n");
			}
			for (int i = 0; i < fullInputs.size(); i++) {
				fw.append(Arrays.toString(fullInputs.get(i)).substring(1, Arrays.toString(fullInputs.get(i)).length()-1) + "\n");
			}
			fw.close();
			// System.out.println("Full DB filled!");
		} catch (IOException e) {
			System.out.println("Error while writing on db");
		}
		return inputs;
	}
	
	protected static String train(String dataPath) {
		List<String[]> inputs;
		
		inputs = fill(dataPath);
		
		// System.out.println("Import model...");
		
		MultiLayerNetwork model = null;
		try {
			model = MultiLayerNetwork.load(new File("model.dl4j"), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("MultiLayerNetwork class not found");
			return "MultiLayerNetwork class not found";
		}
		
		if (inputs == null) {
			return "No data to train was found.";
		}
		
		// Esencialmente crear un dataset como en NetworkTraining
		
		INDArray input = Nd4j.create(new int[]{ inputs.size(), N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ inputs.size(), 320 });
		for (int i = 0; i < inputs.size(); i++) {
			input.putRow(0, Nd4j.createFromArray(new float[] {
				Integer.parseInt(inputs.get(i)[1].substring(1)), 
				Integer.parseInt(inputs.get(i)[2].substring(1)), 
				Integer.parseInt(inputs.get(i)[3].substring(1)),
//				Integer.parseInt(inputs.get(i)[4]),
				Integer.parseInt(inputs.get(i)[5]),
//				Integer.parseInt(inputs.get(i)[6]),
//				Float.parseFloat(inputs.get(i)[7]),
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
		
		int nEpochs = 250;
		model.fit(dsi, nEpochs);
		
		return "ANN trained successfully!";
	}
	
	public static void main(String[] args) {
		fill(localPath + "/tmp");
		return;
	}
}