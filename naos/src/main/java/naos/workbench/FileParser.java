package naos.workbench;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class FileParser {
	String path;
	String fileName;
	File f;
	private static final int INPUTS = 6; // Sin lineas del .c
	
	// En el constructor se divide entre el path del archivo que nos han pasado (la carpeta donde vamos a buscar)
	// y el archivo de donde sacaremos los datos iniciales que usaremos para encontrar el input de la ANN
	public FileParser(String path) throws FileNotFoundException {
		this.path = path.substring(0, path.lastIndexOf("/"));
		this.fileName = path.substring(path.lastIndexOf("/") + 1);
		this.f = new File(this.path);
	}
	
	
	public String[] getInputs() throws FileNotFoundException {
		String[] parts = this.fileName.split("_");
		String[] inputs = new String[INPUTS];
		double minTime = Double.MAX_VALUE;
		
		// Asignamos nombre del programa, #mutantes, #tests, #cores, algoritmo inicial y optimizaciones iniciales
		for (int i = 0; i < INPUTS; i++) {
		      inputs[i] = parts[5+i];
		}
		
		// Recorre todos los ficheros que coinciden en programa, mutantes, tests y cores
		for (final File fileEntry : this.f.listFiles()) {
			String[] tmp = fileEntry.getName().split("_");
			if (inputs[0].equals(tmp[5]) && inputs[1].equals(tmp[6]) && inputs[2].equals(tmp[7]) && inputs[3].equals(tmp[8])) {
				// Si el tiempo del fichero es menor que el que tenemos, actualiza el minimo, algoritmo y optimizaciones
				double tmpTime = getTime(fileEntry.getName());
				if (minTime > tmpTime) {
					minTime = tmpTime;
					inputs[4] = tmp[9];
					inputs[5] = tmp[10];
				}
			}
	    }
		return inputs;
	}

	private double getTime(String fileName) throws FileNotFoundException {
		// Se crea un FILE para buscar exactamente en el fichero que queremos (siempre se llama igual)
		File f = new File(this.path + "/" + fileName + "/malone_overview.txt");
		Scanner scan = new Scanner(f);
		while (scan.hasNextLine()) {
			String[] tmp = scan.nextLine().split(":");
			// Recorremos hasta llegar a la linea que nos interesa y devolvemos el valor del tiempo
			if (tmp[0].equals("Total time")) { // RegEx
				scan.close();
				return Double.parseDouble(tmp[1]);
			}
		}
		scan.close();
		return Double.MAX_VALUE;
	}

	public static void main(String[] args) {
		FileParser fp = null;
		String[] inputs;
		
		try {
			fp = new FileParser(args[0]);
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
		
		for (int i = 0; i < inputs.length; i++) {
			System.out.println("Inputs " + i + ": " + inputs[i]);
		}

		return;
	}
}