#!/bin/bash
###################################################################
##
##             LIMPIAR LOS ARCHIVOS DE LA EJECUCION
##
###################################################################

if [ $# -lt 1 ]
then
    echo "Wrong input command:"
    echo "./clean.sh <fileName>"
else
    rm $MUTOMVO_HOME/apps/$1.c
    rm $MUTOMVO_HOME/apps/$1
    rm -rf $MUTOMVO_HOME/project_$1
    rm $MALONE_HOME/Environments/TFG/$1_stand.ini
fi
