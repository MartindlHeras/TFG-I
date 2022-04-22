import java.io.File;
import java.nio.file.Files;

public class Mutate {
    public static void main(String[] args) {
        if (args.length != 3) {
			System.out.println("Wrong command input, please structure it like: <appName> $MUTOMVO_HOME $MALONE_HOME");
			return;
		}
		System.out.println("####################### SENDING FILES TO MUTOMVO... #######################");
        Files.copy("apps/" + args[1] + "/" + args[1] + ".c", args[2] + "/apps", StandardCopyOption.REPLACE_EXISTING);
        File projectDir = new File(args[2] + "/project_" + args[1]);
        if (!projectDir.exists()){
            projectDir.mkdirs();
        }
        Files.copy("apps/" + args[1] + "/tests_" + args[1] + ".txt", args[2] + "/project_" + args[1], StandardCopyOption.REPLACE_EXISTING);
        Files.copy("comp_remove.sh", args[2] + "/project_" + args[1], StandardCopyOption.REPLACE_EXISTING);


        System.out.println("########################### RUNNING MUTOMVO... ############################");
        // cd $MUTOMVO_HOME
        // ./run_scaled java -jar dist/mutomvo.jar
        // # ./java -jar dist/mutomvo.jar
        // cd -


        System.out.println("########################### ZIPPING MUTANTS ... ###########################");
        // cd $MUTOMVO_HOME/project_$1/mutants
        // zip -r mutants.zip *
        // cd -
        ZipUtil.pack(new File(args[2] + "/project_" + args[1] + "/mutants"), new File("apps/" + args[1] + "/mutants.zip"));
        // File myFile = new File(args[2] + "/project_" + args[1] + "/mutants/mutants.zip");
        // myFile.renameTo(new File("apps/" + args[1] + "/mutants.zip"));


        System.out.println("########################## CREATING AUTOTESTS... ##########################");
        File autotestDir = new File(args[3] + "/Environments/autotest/" + args[1]);
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