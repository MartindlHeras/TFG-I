package naos.workbench;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class Autotest {
	
	protected void singleGenerate(String appName, String tests, String mutomvo, String malone, int cores) {
		try {
			for (int i = 0; i <= 1; i++) {
				String fileName = "test_autotest_" + appName + "_stand_" + i + "_m1_t" + tests + "_w" + cores + ".ini";
				File projectDir = new File(malone + "/Environments/TFG");
				if (!projectDir.exists()) {
					projectDir.mkdirs();
				}
				String fullFileName = malone + "/Environments/TFG/" + fileName;
				FileWriter fw = null;
				try {
					fw = new FileWriter(fullFileName);
				} catch (FileNotFoundException e) {
					return;
				}
				fw.write("[general]"
						+ "\nFrameworkPath=" + mutomvo + ""
						+ "\nApplicationPath=" + mutomvo + "/apps"
						+ "\nMutantPath=" + mutomvo + "/project_" + appName + "/mutants"
						+ "\nApplicationName=" + appName + ""
						+ "\nExecutionLineOriginal=[[MUTANTS_PATH]]/0/"
						+ "\nExecutionLineMutants=[[MUTANTS_PATH]]/[[INDEX_MUTANT]]/"
						+ "\nGenerationLineMutants=cd " + mutomvo + "/bin && java -jar mutomvo.jar -p " + appName + " -g"
						+ "\nTotalTests=" + tests + ""
						+ "\nTotalMutants=1"
						+ "\nStartingMutant=0"
						+ "\n\n[optimizations]"
						+ "\nDistributeOriginal=" + i + ""
						+ "\nSortTestSuite=0"
						+ "\nScatterWorkload=0"
						+ "\nClusterMutants=0"
						+ "\nParallelCompilation=0"
						+ "\nParallelMd5sum=0"
						+ "\nMultipleCoordinators=0"
						+ "\n\n[standalone]"
						+ "\nStandalone=1"
						+ "\nTestSuiteFile=" + mutomvo + "/project_" + appName + "/testsFile.txt"
						+ "\n\n[compilation]"
						+ "\nCompilationEnabled=1"
						+ "\nCompilationLineOriginal=gcc -O3 -lm -Wall [[ORIGINAL_PATH]]/" + appName + ".c -o [[MUTANTS_PATH]]/0/" + appName + ""
						+ "\nCompilationLineMutants=gcc -O3 -lm -Wall [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/" + appName + ".c -o [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/" + appName + ""
						+ "\nCompilationNumWorkers=3"
						+ "\nCompilationWithScript=0"
						+ "\nCompilationScript="
						+ "\n\n[timeouts]"
						+ "\nMALONE_MAX_ORIGINAL_TIMEOUT=90"
						+ "\nMALONE_MAX_MUTANTS_TIMEOUT_FACTOR=5"
						+ "\nMALONE_MAX_MUTANTS_MINIMUM_TIME=17"
						+ "\n\n[monitor]"
						+ "\nMonitorEnabled=0"
						+ "\nMonitorLines=vmstat,ip -s link,top -n 1 -b"
						+ "\nMonitorOnceLines=sysctl -a,lscpu,cat /proc/meminfo"
						+ "\nMonitorFrequency=60"
						+ "\n\n[misc]"
						+ "\nMarkerToken="
						+ "\nMutantGenerationEnabled=0");
				fw.close();
			}
		}
		catch (IOException e) {
			System.out.println("Error writing files");
			e.printStackTrace();
		}
	}
	
	
	protected void generate(String appName, String mutants, String tests, String mutomvo, String malone) {
		boolean bForceCleaningComp;
		
		bForceCleaningComp = true;
		try {
			FileWriter testsW = new FileWriter(malone + "/Environments/autotest/" + appName + "/exec_" + appName + ".sh");
			// System.out.println("Creating autotests files...");
			for (int i = 0; i < 64; i++) {
				String bOpt = String.format("%6s", Integer.toBinaryString(i)).replaceAll(" ", "0");
				String fileName = "test_autotest_" + appName + "_stand_" + bOpt + "_m" + mutants + "_t" + tests + ".ini";
				String fullFileName = malone + "/Environments/autotest/" + appName + "/" + fileName;
				FileWriter fw = new FileWriter(fullFileName);
				fw.write("[general]"
						+ "\nFrameworkPath=" + mutomvo + ""
						+ "\nApplicationPath=" + mutomvo + "/apps"
						+ "\nMutantPath=" + mutomvo + "/project_" + appName + "/mutants"
						+ "\nApplicationName=" + appName + ""
						+ "\nExecutionLineOriginal=[[MUTANTS_PATH]]/0/"
						+ "\nExecutionLineMutants=[[MUTANTS_PATH]]/[[INDEX_MUTANT]]/"
						+ "\nGenerationLineMutants=cd " + mutomvo + "/bin && java -jar mutomvo.jar -p " + appName + " -g"
						+ "\nTotalTests=" + tests + ""
						+ "\nTotalMutants=" + mutants + ""
						+ "\nStartingMutant=1"
						+ "\n\n[optimizations]"
						+ "\nDistributeOriginal=" + bOpt.charAt(0) + ""
						+ "\nSortTestSuite=" + bOpt.charAt(1) + ""
						+ "\nScatterWorkload=" + bOpt.charAt(2) + ""
						+ "\nClusterMutants=" + bOpt.charAt(3) + ""
						+ "\nParallelCompilation=" + bOpt.charAt(4) + ""
						+ "\nParallelMd5sum=" + bOpt.charAt(5) + ""
						+ "\nMultipleCoordinators=0"
						+ "\n\n[standalone]"
						+ "\nStandalone=1"
						+ "\nTestSuiteFile=" + mutomvo + "/project_" + appName + "/testsFile.txt"
						+ "\n\n[compilation]"
						+ "\nCompilationEnabled=1"
						+ "\nCompilationLineOriginal=gcc -O3 -lm -Wall [[ORIGINAL_PATH]]/" + appName + ".c -o [[MUTANTS_PATH]]/0/" + appName + ""
						+ "\nCompilationLineMutants=gcc -O3 -lm -Wall [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/" + appName + ".c -o [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/" + appName + ""
						+ "\nCompilationNumWorkers=3"
						+ "\nCompilationWithScript=0"
						+ "\nCompilationScript="
						+ "\n\n[timeouts]"
						+ "\nMALONE_MAX_ORIGINAL_TIMEOUT=90"
						+ "\nMALONE_MAX_MUTANTS_TIMEOUT_FACTOR=5"
						+ "\nMALONE_MAX_MUTANTS_MINIMUM_TIME=17"
						+ "\n\n[monitor]"
						+ "\nMonitorEnabled=0"
						+ "\nMonitorLines=vmstat,ip -s link,top -n 1 -b"
						+ "\nMonitorOnceLines=sysctl -a,lscpu,cat /proc/meminfo"
						+ "\nMonitorFrequency=60"
						+ "\n\n[misc]"
						+ "\nMarkerToken="
						+ "\nMutantGenerationEnabled=0");
				fw.close();
				
				for (int j = 2; j <= 5; j++) {
					for (int k = 1; k <= 3 ; k++) {
						
						if(bForceCleaningComp)
							testsW.write("sh " + mutomvo + "/project_" + appName + "/comp_remove.sh\n");
						testsW.write("mpirun -n " + (int) Math.pow(2, k) + " ./malone -e " + "autotest/" + appName + "/" + fileName + " -a " + j + "\n");
					}
				}
			}
			testsW.close();
		}
		catch (IOException e) {
			System.out.println("Error writing files");
			e.printStackTrace();
		}
		// System.out.println("Autotest - End");
	}
	
	public static void main(String[] args) {
		Autotest autotest = new Autotest();
		if (args.length < 3) {
			System.out.println("Autotest.java <fileName> <mutants> <tests> <$MUTOMVO_HOME> <$MALONE_HOME> to generate autotests");
			return;
		}
		autotest.generate(args[0], args[1], args[2], args[3], args[4]);
	}

}
