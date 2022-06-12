package naos.workbench;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.Scanner;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

public class Tester {
	private static final int N_INPUTS = 6;
	private static final int N_OUTCOMES = 320;
	private static final String localPath = "/home/martin/Documents/TFG_I/data/";
	
	protected static Double testNetwork(Double pct, Double pctNet, String dbPath) {
		
		DataSet dataSet = null;
		MultiLayerNetwork model = null;
		
		int alg = 0;
		int optInt = 0;
		// CARGAMOS EL PREDICTOR Y LA DB
		try {
			model = MultiLayerNetwork.load(new File("model" + (int)(pctNet*100) + "aug.dl4j"), true);
			dataSet = getDataSet(localPath + dbPath, (int) Files.lines(Paths.get(localPath + dbPath)).count());
		} catch (IOException e) {
			System.out.println("MultiLayerNetwork or DB not found: " + e);
			return -1.0;
		}
		// OBTENEMOS EL ARRAY DE SALIDAS DE LA DB
		INDArray output = model.output(dataSet.getFeatures());
		Scanner testingDBReader;
		Scanner validResultsReader;
		
		double aciertos = 0;
		int n = 0;
		try {
			testingDBReader = new Scanner(new File(localPath + dbPath));
			while (testingDBReader.hasNextLine() && output.getRow(n) != null) {
				String line = testingDBReader.nextLine();
				String[] data = line.split(", ");
				
				float maxO = 0;
				int pos = 0;
				
				for (int i = 0; i < 320; i++) {
					if (output.getRow(n).getFloat(i) > maxO) {
						pos = i;
						maxO = output.getRow(n).getFloat(i);
					}
				}
				alg = (pos/64)+1;
				optInt = pos%64|64;
				
				// LEEMOS EN EL FICHERO DE RESPUESTAS VALIDAS Y SUMAMOS SI ACIERTA
				validResultsReader = new Scanner(new File(localPath + (int)(pct*100) + "/augmentedValidResultsDB" + (int)(pct*100) + ".csv"));
				while (validResultsReader.hasNextLine()) {
					String l = validResultsReader.nextLine();
					String[] parts = l.split(", ");
					if (parts[0].equals(data[0]) && parts[1].equals(data[1]) && parts[2].equals(data[2]) && parts[3].equals(data[3]) && parts[12].equals("a" + alg) && parts[13].equals(Integer.toBinaryString(optInt).substring(1))) {
						aciertos += 1;
						break;
					}
				}
				validResultsReader.close();
				n++;
			}
			testingDBReader.close();
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e);
			return null;
		}
		
		return aciertos/((double) n);
	}

	private static INDArray crearSalida(Integer alg, String opt) {
		int[] out = new int[N_OUTCOMES];
		out[64*(alg-1)+Integer.parseInt(opt,2)] = 1;
		return Nd4j.createFromArray(out);
	}
	
	private static DataSet getDataSet(String folderPath, int nSamples) throws IOException {
		
		INDArray input = Nd4j.create(new int[]{ nSamples, N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ nSamples, N_OUTCOMES });
		
		int n = 0;
		try {
			Scanner testingDBReader = new Scanner(new File(folderPath));
			while (testingDBReader.hasNextLine()) {
				String line = testingDBReader.nextLine();
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
			testingDBReader.close();
	    } catch (FileNotFoundException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
	    }
		
		DataSet dataSet = new DataSet( input, output );
		DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
		return dataSet;
	}
	
	public static void main(String[] args) {
		Double acc = 0.0;
		DecimalFormat df = new DecimalFormat("#.####");
		try {
			FileWriter fw = new FileWriter(new File(localPath + "augmentedPerformance.csv"));
			fw.write("accuracy, 0, 5, 10, 15, 20, 25\n");
			for (int j = 0; j <= 5; j++) {
				fw.write((int)(0.05*j*100) + ", ");
				for (int i = 0; i <= 5 ; i++) {
					acc = testNetwork(0.05*i, 0.05*j, (int)(0.05*i*100) + "/augtesting" + (int)(0.05*i*100) + ".csv");
					System.out.println("Accuracy at " + (int)(0.05*i*100) + ": " + df.format(acc) + "\n");
					if (i != 5) {
						fw.write(df.format(acc) + ", ");
					}
					else {
						fw.write(df.format(acc) + "\n");
					}
				}
			}
			fw.close();
			
		} catch (IOException e) {
			System.out.println("Error: " + e);
		}

		System.out.println("DONE!");
		return;
	}

}
