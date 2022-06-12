package naos.workbench;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;


public class DataAugmenter {
	
	private static final String dbPath = "/home/martin/Documents/TFG_I/data/";
	
	private static List<String[]> extend(Double pct) {
		List<String[]> inputs = new ArrayList<String[]>();
		try {
			Scanner myReader = new Scanner(new File(dbPath + (int)(pct*100) + "/validResultsDB" + (int)(pct*100) + ".csv"));
			myReader.nextLine();
			while (myReader.hasNextLine()) {
				String line = myReader.nextLine();
				String[] input = line.split(", ");
				inputs.add(new String[]{ 
						input[0],
						"m" + Integer.toString(Integer.parseInt(input[1].substring(1))/2), // mutantes
						input[2],
						input[3],
						Integer.toString(Integer.parseInt(input[4])/2), // tiempo compilacion
						input[5],
						Integer.toString(Integer.parseInt(input[6])/2), // tiempo mutantes
						Integer.toString(Integer.parseInt(input[7])/2),
						input[8],
						input[9],
						input[10],
						input[11],
						input[12],
						input[13]});
				inputs.add(new String[]{ 
						input[0],
						"m" + Integer.toString(Integer.parseInt(input[1].substring(1))/3), // mutantes
						input[2],
						input[3],
						Integer.toString(Integer.parseInt(input[4])/3), // tiempo compilacion
						input[5],
						Integer.toString(Integer.parseInt(input[6])/3), // tiempo mutantes
						Integer.toString(Integer.parseInt(input[7])/3),
						input[8],
						input[9],
						input[10],
						input[11],
						input[12],
						input[13]});
				inputs.add(new String[]{ 
						input[0],
						"m" + Integer.toString(Integer.parseInt(input[1].substring(1))*2), // mutantes
						input[2],
						input[3],
						Integer.toString(Integer.parseInt(input[4])*2), // tiempo compilacion
						input[5],
						Integer.toString(Integer.parseInt(input[6])*2), // tiempo mutantes
						Integer.toString(Integer.parseInt(input[7])*2),
						input[8],
						input[9],
						input[10],
						input[11],
						input[12],
						input[13]});
			}
			myReader.close();
	    } catch (FileNotFoundException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
	    }
		
		return inputs;
	}
	
	private static void writeDB(List<String[]> inputs, Double pct) {
				
		if (inputs == null) {
			return;
		}
		
		try {
			FileWriter fw = new FileWriter(dbPath + (int)(pct*100) + "/augmentedValidResultsDB" + (int)(pct*100) + "p.csv", true);
			for (int i = 0; i < inputs.size(); i++) {
				fw.append(Arrays.toString(inputs.get(i)).substring(1, Arrays.toString(inputs.get(i)).length()-1) + "\n");
			}
			fw.close();
		} catch (IOException e) {
			System.out.println("Error while writing on db");
		}
		return;		
	}
	
	private static void getValiddb(Double pct) {
		// input -> [nombre,mutantes,tests,cores,tiempo compilacion,tiempo original,
		// tiempo mutantes,mutation score,lineas c,size TS,algoritmo,optimizaciones]
		if (pct == 0.0) {
			try {
				Files.copy((new File(dbPath + "fullshortdb.csv")).toPath(),
						(new File(dbPath + (int)(pct*100) + "/validResultsDB" + (int)(pct*100) + ".csv")).toPath(), StandardCopyOption.REPLACE_EXISTING);
			} catch (IOException e) {
				System.out.println("File not found\n");
				System.out.println(e);
				return;
			}
			return;
		}
		
		try {
			Scanner myReader = new Scanner(new File(dbPath + "fullshortdb.csv"));
			myReader.nextLine();
			FileWriter fw = new FileWriter(dbPath + (int)(pct*100) + "/validResultsDB" + (int)(pct*100) + ".csv", true);
			while (myReader.hasNextLine()) {
				String line = myReader.nextLine();
				String[] input = line.split(", ");
				Scanner fullReader = new Scanner(new File(dbPath + "fulldb.csv"));
				fullReader.nextLine();
				
				while (fullReader.hasNextLine()) {
					String l = fullReader.nextLine();
					String[] parts = l.split(", ");
					if (Double.parseDouble(parts[7]) <= (1+pct)*Double.parseDouble(input[7]) && parts[0].equals(input[0]) && parts[1].equals(input[1]) && parts[2].equals(input[2]) && parts[3].equals(input[3]) && parts[9].equals(input[9]) && parts[10].equals(input[10]) && parts[11].equals(input[11])) {
						fw.append(l + "\n");
					}
				}
				fullReader.close();
			}
			fw.close();
			myReader.close();
	    } catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
	    }
		
		return;
	}

	public static void main(String[] args) {
		Double pct = 0.0;
		writeDB(extend(pct), pct);
		getValiddb(pct);
		System.out.println("DONE!\n");
		return;
	}
}
