package naos.workbench;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class Mutate {
	
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
	
	public static void main(String[] args) {
		if (args.length != 4) {
			System.out.println("Wrong command input, please structure it like:");
			System.out.println("<appName> $MUTOMVO_HOME $MALONE_HOME <localPath>");
			return;
		}
		String app = args[0];
		String mutomvo = args[1];
		String malone = args[2];
		String local = args[3];
		int mutants = 0;
		System.out.println("####################### SENDING FILES TO MUTOMVO... #######################");
		// Copy app.c to mutomvo
		try {
			Files.copy((new File(local + "/apps/" + app + "/" + app + ".c")).toPath(),
					(new File(mutomvo + "/apps/" + app + ".c")).toPath(), StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + local + "/apps/" + app + "/" + app + ".c not found");
			System.out.println(e);
			return;
		}
		// Create project_app in mutomvo
		File projectDir = new File(mutomvo + "/project_" + app);
		if (!projectDir.exists()) {
			projectDir.mkdirs();
		}
		// Copy tests_app.txt to mutomvo
		try {
			Files.copy((new File(local + "/apps/" + app + "/tests_" + app + ".txt")).toPath(),
					(new File(mutomvo + "/project_" + app + "/tests_" + app + ".txt")).toPath(),
					StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + local + "/apps/" + app + "/tests_" + app + ".txt not found");
			System.out.println(e);
			return;
		}
		// Copy comp_remove.sh to mutomvo
		try {
			Files.copy((new File(local + "/scripts/comp_remove.sh")).toPath(),
					(new File(mutomvo + "/project_" + app + "/comp_remove.sh")).toPath(),
					StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + local + "/scripts/comp_remove.sh not found");
			System.out.println(e);
			return;
		}
		
		mutants = new File(mutomvo + "/project_" + app + "/mutants").list().length;
		if (mutants == 0) {
			System.out.println("########################### RUNNING MUTOMVO... ############################");
			try {
				Process p = Runtime.getRuntime().exec("run_scaled java -jar " + mutomvo + "/dist/mutomvo.jar"); // This for DPI screens
//    			Process p = Runtime.getRuntime().exec("java -jar " + mutomvo + "/dist/mutomvo.jar"); // This for normal screens
				p.waitFor();
			} catch (IOException | InterruptedException e) {
				System.out.println("######################## ERROR RUNNING MUTOMVO... #########################");
				System.out.println(e);
			}
			
			System.out.println("########################### ZIPPING MUTANTS ... ###########################");
			try {
				String sourceFile = mutomvo + "/project_" + app + "/mutants";
				FileOutputStream fos = new FileOutputStream(local + "/apps/" + app + "/mutants.zip");
				ZipOutputStream zipOut = new ZipOutputStream(fos);
				File fileToZip = new File(sourceFile);

				zipFile(fileToZip, fileToZip.getName(), zipOut);
				zipOut.close();
				fos.close();
			} catch (IOException e) {
				System.out.println("######################## ERROR ZIPPING MUTANTS ... ########################");
				System.out.println(e);
			}
		}

		System.out.println("########################## CREATING AUTOTESTS... ##########################");
		File autotestDir = new File(malone + "/Environments/autotest/" + app);
		if (!autotestDir.exists()) {
			autotestDir.mkdirs();
		}
		Autotest autotest = new Autotest();
		BufferedReader reader = null;
		int tests = 0;
		try {
			reader = new BufferedReader(new FileReader(local + "/apps/" + app + "/tests_" + app + ".txt"));
			while (reader.readLine() != null)
				tests++;
			reader.close();
		} catch (IOException e) {
			System.out.println("####################### ERROR CREATING AUTOTESTS... #######################");
			System.out.println(e);
			return;
		}
		autotest.generate(app, String.valueOf(mutants), String.valueOf(tests), mutomvo, malone);
		
		System.out.println("################################## DONE! ##################################");
	}
}