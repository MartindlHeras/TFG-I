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
    cd $MUTOMVO_HOME
    ./run_scaled java -jar dist/mutomvo.jar
    # ./java -jar dist/mutomvo.jar
    cd -

    echo "########################### ZIPPING MUTANTS ... ###########################"
    cd $MUTOMVO_HOME/project_$1/mutants
    zip -r mutants.zip *
    cd -
    mv $MUTOMVO_HOME/project_$1/mutants/mutants.zip apps/$1/

    echo "########################## CREATING AUTOTESTS... ##########################"

    mkdir $MALONE_HOME/Environments/autotest/$1

    cd naos/src/main/java/naos/workbench
    javac Autotest.java
    java Autotest.java $1 $(ls $MUTOMVO_HOME/project_$1/mutants/ | wc -l) $(sed -n "$=" ../../../../../../apps/$1/tests_$1.txt) $MUTOMVO_HOME $MALONE_HOME
    cd -

    echo "################################## DONE! ##################################"
fi
