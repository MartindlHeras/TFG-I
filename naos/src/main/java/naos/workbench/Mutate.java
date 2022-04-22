package naos.workbench;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import org.apache.commons.compress.archivers.zip.ZipUtil;

public class Mutate {
    public static void main(String[] args) {
        if (args.length != 4) {
			System.out.println("Wrong command input, please structure it like: <appName> $MUTOMVO_HOME $MALONE_HOME <localPath>");
			return;
		}
        String app = args[0];
        String mutomvo = args[1];
        String malone = args[2];
        String local = args[3];
		System.out.println("####################### SENDING FILES TO MUTOMVO... #######################");
        try {
			Files.copy((new File(local + "/apps/" + app + "/" + app + ".c")).toPath(), (new File(mutomvo + "/apps/"+ app + ".c")).toPath(), StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + local + "/apps/" + app + "/" + app + ".c not found");
			System.out.println(e);
			return;
		}

        File projectDir = new File(mutomvo + "/project_" + app);
        if (!projectDir.exists()){
            projectDir.mkdirs();
        }
        try {
			Files.copy((new File(local + "/apps/" + app + "/tests_" + app + ".txt")).toPath(),(new File(mutomvo + "/project_" + app + "/tests_" + app + ".txt")).toPath(), StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + local + "/apps/" + app + "/tests_" + app + ".txt not found");
			System.out.println(e);
			return;
		}
        try {
			Files.copy((new File(local + "/comp_remove.sh")).toPath(), (new File(mutomvo + "/project_" + app + "/comp_remove.sh")).toPath(), StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			System.out.println("File " + local + "/comp_remove.sh not found");
			System.out.println(e);
			return;
		}


        System.out.println("########################### RUNNING MUTOMVO... ############################");
        // cd $MUTOMVO_HOME
        // ./run_scaled java -jar dist/mutomvo.jar
        // # ./java -jar dist/mutomvo.jar
        // cd -


        System.out.println("########################### ZIPPING MUTANTS ... ###########################");
        // cd $MUTOMVO_HOME/project_$1/mutants
        // zip -r mutants.zip *
        // cd -
//        (new File(mutomvo + "/project_" + app + "/mutants/mutants.zip")).renameTo(new File(local + "apps/" + app + "/mutants.zip"));


        System.out.println("########################## CREATING AUTOTESTS... ##########################");
        File autotestDir = new File(malone + "/Environments/autotest/" + app);
        if (!autotestDir.exists()){
            autotestDir.mkdirs();
        }

        // cd naos/src/main/java/naos/workbench
        // javac Autotest.java
        // java Autotest.java $1 $(ls $MUTOMVO_HOME/project_$1/mutants/ | wc -l) $(sed -n "$=" ../../../../../../apps/$1/tests_$1.txt) $MUTOMVO_HOME $MALONE_HOME
        // # java -jar Autotest.jar $1 $(ls $MUTOMVO_HOME/project_$1/mutants/ | wc -l) $(sed -n "$=" ../../../../../../apps/$1/tests_$1.txt) $MUTOMVO_HOME $MALONE_HOME
        // cd -

        System.out.println("################################## DONE! ##################################");
    }
}