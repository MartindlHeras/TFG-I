package naos.workbench;

import java.io.FileWriter;
import java.io.IOException;

public class Autotest {

	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("Autotest.java <fileName> <mutants> <tests> <$MUTOMVO_HOME> <$MALONE_HOME> to generate autotests");
			return;
		}
		try {
			FileWriter testsW = new FileWriter(args[4] + "/Environments/autotest/" + args[0] + "/execLines.txt");
			for (int i = 0; i < 64; i++) {
				String bOpt = String.format("%6s", Integer.toBinaryString(i)).replaceAll(" ", "0");
				String fileName = "test_autotest_" + args[0] + "_stand_" + bOpt + "_m" + args[1] + "_t" + args[2] + ".ini";
				System.out.println("Creating file: " + fileName + "...");
				String fullFileName = args[4] + "/Environments/autotest/" + args[0] + "/" + fileName;
//				FileWriter fw = new FileWriter(args[4] + "/Environments/autotest/" + fileName);
				FileWriter fw = new FileWriter(fullFileName);
				fw.write("[general] "
						+ "\nFrameworkPath=" + args[3] + ""
						+ "\nApplicationPath=" + args[3] + "/apps"
						+ "\nMutantPath=" + args[3] + "/project_" + args[0] + "/mutants"
						+ "\nApplicationName=" + args[0] + ""
						+ "\nExecutionLineOriginal=[[ORIGINAL_PATH]]/"
						+ "\nExecutionLineMutants=[[MUTANTS_PATH]]/[[INDEX_MUTANT]]/"
						+ "\nGenerationLineMutants=cd " + args[3] + "/bin && java -jar mutomvo.jar -p " + args[0] + " -g"
						+ "\nTotalTests=" + args[2] + ""
						+ "\nTotalMutants=" + args[1] + ""
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
						+ "\nTestSuiteFile=" + args[3] + "/project_" + args[0] + "/testsFile.txt"
						+ "\n\n[compilation]"
						+ "\nCompilationEnabled=1"
						+ "\nCompilationLineOriginal=gcc -O3 -lm -Wall [[ORIGINAL_PATH]]/" + args[0] + ".c -o [[ORIGINAL_PATH]]/" + args[0] + " "
						+ "\nCompilationLineMutants=gcc -O3 -lm -Wall [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/" + args[0] + ".c -o [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/" + args[0] + ""
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
				for (int j = 1; j <= 5; j++) {
					for (int k = 1; k <= 5 ; k++) {
						testsW.write("mpirun -n " + (int) Math.pow(2, k) + " ./malone -e " + "autotest/" + args[0] + "/" + fileName + " -a " + j + "\n");
					}
				}
				}
			testsW.close();
		}
		catch (IOException e) {
			System.out.println("Error writing files");
			e.printStackTrace();
		}

	}

}