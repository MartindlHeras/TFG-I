#!/bin/bash
###################################################################
##
##               SCRIPT DE EJECUCION DE FICHEROS .c
##
###################################################################

if [ $# -lt 1 ]
then
    echo "Wrong input command:"
    echo "./mutate.sh <appName>"
else
    echo "####################### SENDING FILES TO MUTOMVO... #######################"
    cp apps/$1/$1.c $MUTOMVO_HOME/apps
    mkdir $MUTOMVO_HOME/project_$1
    cp apps/$1/tests_$1.txt $MUTOMVO_HOME/project_$1

    echo "########################### RUNNING MUTOMVO... ############################"
    # ./$MUTOMVO_HOME/run_scaled java -jar $MUTOMVO_HOME/dist/mutomvo.jar
    cd $MUTOMVO_HOME
    ./run_scaled java -jar dist/mutomvo.jar
    cd -

    echo "########################### ZIPPING MUTANTS ... ###########################"
    cd $MUTOMVO_HOME/project_$1/mutants
    zip -r mutants.zip *
    cd -
    mv $MUTOMVO_HOME/project_$1/mutants/mutants.zip apps/$1/

    echo "####################### CREATING stand.ini FILE... ########################"
    echo '[general]
    FrameworkPath=/localStorage/mutomvo
    ApplicationPath=/localStorage/mutomvo/apps
    MutantPath=/localStorage/mutomvo/project_'$1'/mutants
    ApplicationName='$1'
    ExecutionLineOriginal=[[ORIGINAL_PATH]]/
    ExecutionLineMutants=[[MUTANTS_PATH]]/[[INDEX_MUTANT]]/
    GenerationLineMutants=cd /localStorage/mutomvo/bin && java -jar mutomvo.jar -p '$1' -g
    TotalTests='$(sed -n "$=" apps/$1/tests_$1.txt)'
    TotalMutants='$(ls $MUTOMVO_HOME/project_$1/mutants/ | wc -l)'
    StartingMutant=1

    [optimizations]
    DistributeOriginal=0
    SortTestSuite=0
    ScatterWorkload=0
    ClusterMutants=0
    ParallelCompilation=0
    ParallelMd5sum=0
    MultipleCoordinators=0

    [standalone]
    Standalone=1
    TestSuiteFile=/localStorage/mutomvo/project_'$1'/testsFile.txt

    [compilation]
    CompilationEnabled=1
    CompilationLineOriginal=gcc -O3 -lm -Wall [[ORIGINAL_PATH]]/'$1'.c -o [[ORIGINAL_PATH]]/'$1' 
    CompilationLineMutants=gcc -O3 -lm -Wall [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/'$1'.c -o [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/'$1'
    CompilationNumWorkers=3
    CompilationWithScript=0
    CompilationScript=

    [timeouts]
    MALONE_MAX_ORIGINAL_TIMEOUT=90
    MALONE_MAX_MUTANTS_TIMEOUT_FACTOR=5
    MALONE_MAX_MUTANTS_MINIMUM_TIME=17

    [monitor]
    MonitorEnabled=0
    MonitorLines=vmstat,ip -s link,top -n 1 -b
    MonitorOnceLines=sysctl -a,lscpu,cat /proc/meminfo
    MonitorFrequency=60

    [misc]
    MarkerToken=
    MutantGenerationEnabled=0
    ' > $MALONE_HOME/Environments/TFG/$1_stand.ini

    echo "######################### EXECUTING IN MALONE... ##########################"
    # mpirun -n 2 ./$MALONE_HOME/malone -e $MALONE_HOME/TFG/$1_stand.ini -a 4
    cd $MALONE_HOME
    # echo "######################## EXECUTING ALGORITHM 1... #########################"
    # mpirun -n 2 ./malone -e TFG/$1_stand.ini -a 1
    # echo "######################## EXECUTING ALGORITHM 2... #########################"
    # mpirun -n 2 ./malone -e TFG/$1_stand.ini -a 2
    # echo "######################## EXECUTING ALGORITHM 3... #########################"
    # mpirun -n 2 ./malone -e TFG/$1_stand.ini -a 3
    echo "######################## EXECUTING ALGORITHM 4... #########################"
    mpirun -n 2 ./malone -e TFG/$1_stand.ini -a 4
    # echo "######################## EXECUTING ALGORITHM 5... #########################"
    # mpirun -n 2 ./malone -e TFG/$1_stand.ini -a 5
    cd -

    echo "################################## DONE! ##################################"

    echo "########################## CREATING AUTOTESTS... ##########################"
    # cd naos/src/main/java/naos/workbench
    # javac Autotest.java
    # java Autotest.java $1 $(ls $MUTOMVO_HOME/project_$1/mutants/ | wc -l) $(sed -n "$=" apps/$1/tests_$1.txt)
    # cd -

    echo "################################## DONE! ##################################"

    # mpirun -n 2 ./malone -e autotest/test_autotest_add_stand.ini -a 4
fi
