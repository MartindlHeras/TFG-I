package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FileParser {
	String path;
	String appsPath;
	File f;
	private static final int INPUTS = 12;
	
	// En el constructor se divide entre el path del archivo que nos han pasado (la carpeta donde vamos a buscar)
	// y el archivo de donde sacaremos los datos iniciales que usaremos para encontrar el input de la ANN
	public FileParser(String path, String appsPath) throws FileNotFoundException {
		this.path = path;
		this.appsPath = appsPath;
		this.f = new File(this.path);
	}
	
	
	public List<String[]> getInputs() throws FileNotFoundException {
		List<String[]> inputs = new ArrayList<String[]>();
		
		// input -> [nombre,mutantes,tests,cores,tiempo,tiempo original,tiempo mutantes,mutation score,lineas c,size TS,algoritmo,optimizaciones]
				
		for (final File fileName : this.f.listFiles()) {
			String[] input = new String[INPUTS];
			double minTime = Double.MAX_VALUE;
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
				if (input[0].equals(inputs.get(i)[0]) && input[1].equals(inputs.get(i)[1]) && input[2].equals(inputs.get(i)[2]) && input[3].equals(inputs.get(i)[3])) { // Si ya he pasado por esas especificaciones
					flag = true;
					break;
				}
			}
			if (flag) {
				continue;
			}
			// Recorre todos los ficheros que coinciden en programa, mutantes, tests y cores
			for (final File fileEntry : this.f.listFiles()) {
				String[] tmp = fileEntry.getName().split("_");
				
				if (input[0].equals(tmp[5]) && input[1].equals(tmp[6]) && input[2].equals(tmp[7]) && input[3].equals(tmp[8])) {
					// Si el tiempo del fichero es menor que el que tenemos, actualiza el minimo, algoritmo y optimizaciones
					File f = new File(this.path + "/" + fileEntry.getName() + "/malone_overview.txt");
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
					if (minTime > totalTime) {
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
			Path cPath = Paths.get(this.appsPath + "/" + input[0] + "/" + input[0]+ ".c");
			Path TSPath = Paths.get(this.appsPath + "/" + input[0] + "/tests_" + input[0] + ".txt");
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
	
	public List<String[]> getFullInputs() throws FileNotFoundException {
		List<String[]> inputs = new ArrayList<String[]>();
		
		// input -> [nombre,mutantes,tests,cores,tiempo,tiempo original,tiempo mutantes,mutation score,lineas c,size TS,algoritmo,optimizaciones]
				
		for (final File fileName : this.f.listFiles()) {
			String[] input = new String[INPUTS];
			String[] parts = fileName.getName().split("_");
			// Asignamos nombre del programa, #mutantes, #tests, #cores, algoritmo inicial y optimizaciones iniciales
			for (int i = 0; i < 4; i++) {
			      input[i] = parts[5+i];
			}
			input[10] = parts[9]; // Algoritmo inicial
			input[11] = parts[10]; // Optimizaciones
			
			File f = new File(this.path + "/" + fileName.getName() + "/malone_overview.txt");
			Scanner scan = new Scanner(f);
			while (scan.hasNextLine()) {
				String[] tmpFileTime = scan.nextLine().split(":");
				// Recorremos hasta llegar a la linea que nos interesa y devolvemos el valor del tiempo
				if (tmpFileTime[0].equals("Total time")) { // RegEx
					input[4] = tmpFileTime[1].substring(1);
				}
				if (tmpFileTime[0].equals("Original time")) { // RegEx
					input[5] = tmpFileTime[1].substring(1);
				}
				if (tmpFileTime[0].equals("Mutants time")) { // RegEx
					input[6] = tmpFileTime[1].substring(1);
				}
				if (tmpFileTime[0].equals("Mutation score")) { // RegEx
					input[7] = tmpFileTime[1].substring(1);
				}
			}
			scan.close();			
		    
			// appsFolder = /home/martin/Documents/TFG_I/apps
			Path cPath = Paths.get(this.appsPath + "/" + input[0] + "/" + input[0]+ ".c");
			Path TSPath = Paths.get(this.appsPath + "/" + input[0] + "/tests_" + input[0] + ".txt");
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

	public static void main(String[] args) {
		FileParser fp = null;
		List<String[]> inputs;
		
		try {
			fp = new FileParser(args[0], args[1]);
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e + " while creating object FileParser");
			return;
		}
		
		try {
			inputs = fp.getInputs();
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e + " while getting inputs");
			return;
		}
		
		for (int i = 0; i < inputs.size(); i++) {
			for (int j = 0; j < inputs.get(i).length; j++) {
				System.out.println("Inputs " + i + ":" + j + " = " + inputs.get(i)[j]);
			}
		}

		return;
	}
}