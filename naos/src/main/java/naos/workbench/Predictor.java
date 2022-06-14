package naos.workbench;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.ProcessBuilder.Redirect;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.Scanner;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

public class Predictor {
	
	private static final String localPath = "/home/martin/Documents/TFG_I";
	private static final int N_INPUTS = 6; // 9 si meto los que faltan
	private static final int N_OUTCOMES = 320;
	
	private static void zipFile(File fileToZip, String fileName, ZipOutputStream zipOut) throws IOException {
		if (fileToZip.isHidden()) {
			return;
		}
		if (fileToZip.isDirectory()) {
			if (fileName.endsWith("/")) {
				zipOut.putNextEntry(new ZipEntry(fileName));
				zipOut.closeEntry();
			} else {
				zipOut.putNextEntry(new ZipEntry(fileName + "/"));
				zipOut.closeEntry();
			}
			File[] children = fileToZip.listFiles();
			for (File childFile : children) {
				zipFile(childFile, fileName + "/" + childFile.getName(), zipOut);
			}
			return;
		}
		FileInputStream fis = new FileInputStream(fileToZip);
		ZipEntry zipEntry = new ZipEntry(fileName);
		zipOut.putNextEntry(zipEntry);
		byte[] bytes = new byte[1024];
		int length;
		while ((length = fis.read(bytes)) >= 0) {
			zipOut.write(bytes, 0, length);
		}
		fis.close();
	}
	
	protected static String mutate(String app, String mutomvo, String malone) {
		int mutants = 0;
		// System.out.println("####################### SENDING FILES TO MUTOMVO... #######################");
		// Copy app.c to mutomvo
		try {
			Files.copy((new File(localPath + "/apps/" + app + "/" + app + ".c")).toPath(),
					(new File(mutomvo + "/apps/" + app + ".c")).toPath(), StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + localPath + "/apps/" + app + "/" + app + ".c not found");
			System.out.println(e);
			return "File " + localPath + "/apps/" + app + "/" + app + ".c not found";
		}
		// Create project_app in mutomvo
		File projectDir = new File(mutomvo + "/project_" + app);
		if (!projectDir.exists()) {
			projectDir.mkdirs();
		}
		// Copy tests_app.txt to mutomvo
		try {
			Files.copy((new File(localPath + "/apps/" + app + "/tests_" + app + ".txt")).toPath(),
					(new File(mutomvo + "/project_" + app + "/tests_" + app + ".txt")).toPath(),
					StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + localPath + "/apps/" + app + "/tests_" + app + ".txt not found");
			System.out.println(e);
			return "File " + localPath + "/apps/" + app + "/tests_" + app + ".txt not found";
		}
		// Copy comp_remove.sh to mutomvo
		try {
			Files.copy((new File(localPath + "/scripts/comp_remove.sh")).toPath(),
					(new File(mutomvo + "/project_" + app + "/comp_remove.sh")).toPath(),
					StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + localPath + "/scripts/comp_remove.sh not found");
			System.out.println(e);
			return "File " + localPath + "/scripts/comp_remove.sh not found";
		}
		
		if (new File(mutomvo + "/project_" + app + "/mutants").list() == null) {
			// System.out.println("########################### RUNNING MUTOMVO... ############################");
			try {
//				Process p = Runtime.getRuntime().exec("run_scaled java -jar " + mutomvo + "/dist/mutomvo.jar"); // This for DPI screens
    			Process p = Runtime.getRuntime().exec("java -jar " + mutomvo + "/dist/mutomvo.jar"); // This for normal screens
				p.waitFor();
			} catch (IOException | InterruptedException e) {
				System.out.println("######################## ERROR RUNNING MUTOMVO... #########################");
				System.out.println(e);
				return "Error running Mutomvo";
			}
			
			// System.out.println("########################### ZIPPING MUTANTS ... ###########################");
			try {
				String sourceFile = mutomvo + "/project_" + app + "/mutants";
				FileOutputStream fos = new FileOutputStream(localPath + "/apps/" + app + "/mutants.zip");
				ZipOutputStream zipOut = new ZipOutputStream(fos);
				File fileToZip = new File(sourceFile);

				zipFile(fileToZip, fileToZip.getName(), zipOut);
				zipOut.close();
				fos.close();
			} catch (IOException e) {
				System.out.println("######################## ERROR ZIPPING MUTANTS ... ########################");
				System.out.println(e);
				return "Error zipping mutants";
			}
		}
		mutants = new File(mutomvo + "/project_" + app + "/mutants").list().length;

		// System.out.println("########################## CREATING AUTOTESTS... ##########################");
		File autotestDir = new File(malone + "/Environments/autotest/" + app);
		if (!autotestDir.exists()) {
			autotestDir.mkdirs();
		}
		Autotest autotest = new Autotest();
		BufferedReader reader = null;
		int tests = 0;
		try {
			reader = new BufferedReader(new FileReader(localPath + "/apps/" + app + "/tests_" + app + ".txt"));
			while (reader.readLine() != null)
				tests++;
			reader.close();
		} catch (IOException e) {
			System.out.println("####################### ERROR CREATING AUTOTESTS... #######################");
			System.out.println(e);
			return "Error creating autotests";
		}
		autotest.generate(app, String.valueOf(mutants), String.valueOf(tests), mutomvo, malone);
		
		// System.out.println("################################## DONE! ##################################");
		return app + " mutated successfully!";
	}
	
	protected static String predict(String app, String mutomvo, String malone, String coresText) {
		int mutants;
		Long tests;
		Long tsSize;
		Long lines;
		int cores = 0;
		String returnString = "<html>The optimal execution is:<br/>";
		
		mutate(app, mutomvo, malone);
		
		Path cPath = Paths.get(localPath + "/apps/" + app + "/" + app + ".c");
		Path tsPath = Paths.get(localPath + "/apps/" + app + "/tests_" + app + ".txt");
		try {
			mutants = new File(mutomvo + "/project_" + app + "/mutants").list().length;
			tsSize = Files.size(tsPath); // TAMANO TS
			tests = Files.lines(tsPath).count(); // NUMERO DE TESTS
			lines = Files.lines(cPath).count(); // NUMERO DE LINEAS
		} catch (IOException | NullPointerException e) {
			System.out.println(e);
			return "Error getting parameters.";
		}
		
		try {
			cores = Integer.parseInt(coresText);
		} catch (NumberFormatException | NullPointerException e) {
			cores = Runtime.getRuntime().availableProcessors();
		}
		
		/// EJECUCION MALONE
		Autotest autotest = new Autotest();
		autotest.singleGenerate(app, Long.toString(tests), mutomvo, malone, cores);
		String fileName = "test_autotest_" + app + "_stand_0_m1_t" + tests + "_w" + cores + ".ini";
		String fileNameParalell = "test_autotest_" + app + "_stand_1_m1_t" + tests + "_w" + cores + ".ini";
		String originalTime;
		String originalTimeParalell;
		try {
			FileWriter fw = new FileWriter(malone + "/tmp.sh");
			fw.write("mpirun -n " + cores + " ./malone -e TFG/" + fileName + " -a 4\n"
					+"mpirun -n " + cores + " ./malone -e TFG/" + fileNameParalell + " -a 4\n"
					+"rm -rf tmp.sh");
			fw.close();
			ProcessBuilder builder = new ProcessBuilder("sh","tmp.sh");
			Map<String, String> env = builder.environment();
			env.put("MALONE_HOME", malone);
			env.put("MUTOMVO_HOME", mutomvo);
			builder.directory(new File(malone));
			builder.redirectOutput(Redirect.DISCARD);
			Process p = builder.start();
			p.waitFor();
		} catch (IOException | InterruptedException e) {
			System.out.println(e);
			return "Error running Malone";
		}
		
		originalTime = getOriginalTime(malone + "/Results", fileName);
		originalTimeParalell = getOriginalTime(malone + "/Results", fileNameParalell);
		
		if (originalTime == null || originalTimeParalell == null) {
			return "Error getting time";
		}
		
		// System.out.println("Import model....");
		
		MultiLayerNetwork model = null;
		try {
			model = MultiLayerNetwork.load(new File("model0aug.dl4j"), true);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("MultiLayerNetwork class not found");
			return "Model not found.";
		}

		String dbPath = localPath + "/data/fullshortdb.csv";
		int nSamples = 0;
		try {
			nSamples = (int) Files.lines(Paths.get(localPath + "/data/fullshortdb.csv")).count()-1;
		} catch (IOException e1) {
			System.out.println("Error: " + e1);
			return "Database error.";
		}
		INDArray input = Nd4j.create(new int[]{ nSamples + 2, N_INPUTS });
		INDArray output = Nd4j.create(new int[]{ nSamples + 2, N_OUTCOMES });
		
		int n = 2;
		input.putRow(0, Nd4j.createFromArray( new float[]{
				mutants,
				tests,
				cores,
				Long.parseLong(originalTime),
				tsSize,
				lines
				} ));
		output.putRow(0, crearSalida(4, "000000"));
		
		input.putRow(1, Nd4j.createFromArray( new float[]{
				mutants,
				tests,
				cores,
				Long.parseLong(originalTimeParalell),
				tsSize,
				lines
				} ));
		output.putRow(1, crearSalida(4, "100000"));
		try {
			Scanner testingDBReader = new Scanner(new File(dbPath));
			testingDBReader.nextLine();
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
		
		float maxO = 0;
		int pos = 0;
		INDArray predictions = model.output(dataSet.getFeatures());
		for (int i = 0; i < 320; i++) {
			if (predictions.getRow(0).getFloat(i) > maxO) {
				pos = i;
				maxO = output.getFloat(i);
			}
		}
		int alg = (pos/64)+1;
		int optInt = pos%64|64;
		
		returnString += "single: a" + alg + ", " + Integer.toBinaryString(optInt).substring(1) + "<br/>";
		
		maxO = 0;
		pos = 0;
		for (int i = 0; i < 320; i++) {
			if (predictions.getRow(1).getFloat(i) > maxO) {
				pos = i;
				maxO = output.getFloat(i);
			}
		}
		alg = (pos/64)+1;
		optInt = pos%64|64;
		
		return returnString + "paralell: a" + alg + ", " + Integer.toBinaryString(optInt).substring(1) + "</html>";
	}
	
	private static INDArray crearSalida(Integer alg, String opt) {
		int[] out = new int[N_OUTCOMES];
		out[64*(alg-1)+Integer.parseInt(opt,2)] = 1;
		return Nd4j.createFromArray(out);
	}
	
	private static String getOriginalTime(String dataPath, String fileName) {
		File dataDirectory = new File(dataPath);
		String originalTime = null;
		String[] parts = fileName.split("[_.]");
		if (!dataDirectory.exists()) {
			return null;
		}
		for (final File faux : dataDirectory.listFiles()) {
			String[] partsAux = faux.getName().split("_");
			// Asignamos nombre del programa, #mutantes, #tests, #cores, algoritmo inicial y optimizaciones iniciales
			if (partsAux[5].equals(parts[2]) && partsAux[6].equals(parts[5]) && partsAux[7].equals(parts[6]) && partsAux[8].equals(parts[7]) && partsAux[10].equals(parts[4] + "00000")) {
				File f = new File(dataPath + "/" + faux.getName() + "/malone_overview.txt");
				Scanner scan;
				try {
					scan = new Scanner(f);
					while (scan.hasNextLine()) {
						String[] tmpFileTime = scan.nextLine().split(":");
						if (tmpFileTime[0].equals("Original time")) {
							originalTime = tmpFileTime[1].substring(1);
							break;
						}
					}
					scan.close();
				} catch (FileNotFoundException e) {
					return null;
				}
				break;
			}
		}
		return originalTime;
	}

	public static void main(String[] args) {
//		if (args.length != 3) {
//			System.out.println("Wrong command input, please structure it like:");
//			System.out.println("<appName> $MUTOMVO_HOME $MALONE_HOME");
//			return;
//		}
		System.out.println(predict("add", "/localStorage/mutomvo", "/home/martin/Documents/Malone", "4"));
	}
}