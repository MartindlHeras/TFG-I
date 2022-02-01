package naos.workbench;
import java.io.FileNotFoundException;

public class Naos {
	
	private static void train(String fullFolderName) {
		FileParser fp = null;
		String[] inputs;
		
		// COMPROBAR ARGUMENTOS DE ENTRADA QUE TENGAN SENTIDO
		
		try {
			fp = new FileParser(fullFolderName);
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
		
		// Actualizar valores de la ANN (mas dificil de lo que parece)
		
		return;
	}
	
	private static String predict(String fileName, String nMutants, String nTests, String nCores) {
		
		// COMPROBAR ARGUMENTOS DE ENTRADA QUE TENGAN SENTIDO
		
		// Llamar a la ANN y pasarle los datos (repurpose test/ANN)
		// Devolver algoritmo
		return "a0";
	}
	
	public static void main(String[] args) {
		if (args.length == 0) {
			System.out.println("Wrong command input, please select an option:");
			System.out.println("-t <fullFolderName> to train the Neural Network");
			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> to get the optimal execution mode");
			return;
		}
		switch (args[0]) {
		case "-t":
			train(args[1]);
			System.out.println("ANN trained successfully");
			break;
		case "-p":
			System.out.println("The optimal algorithm for the situation is: " + predict(args[1], args[2], args[3], args[4]));
			break;
		default:
			System.out.println("Wrong command input, please select an option:");
			System.out.println("-t <fullFolderName> to train the Neural Network"); // Tengo que transformar para que funcione tambien con paths locales
			System.out.println("-p <fileName> <nMutants> <nTests> <nCores> to get the optimal execution mode"); // Buscar un modo con menos inputs
			break;
		}
		return;
	}
}
