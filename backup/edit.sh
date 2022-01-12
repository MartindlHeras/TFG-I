###################################################################
##
##          DOS MANERAS DE EDITAR stand.ini Y QUE CAMBIAR          
##
###################################################################

echo "Esta todo comentado"

############################ MANERA 1 #############################

# sed 's/tfgheras/'"${1%.*}"'/g' stand.ini > "${1%.*}"_stand.ini

############################ MANERA 2 #############################

# echo '[general]
# FrameworkPath=/localStorage/mutomvo
# ApplicationPath=/localStorage/mutomvo/apps
# MutantPath=/localStorage/mutomvo/project_'"${1%.*}"'/mutants
# ApplicationName='"${1%.*}"'
# ExecutionLineOriginal=[[ORIGINAL_PATH]]/
# ExecutionLineMutants=[[MUTANTS_PATH]]/[[INDEX_MUTANT]]/
# GenerationLineMutants=cd /localStorage/mutomvo/bin && java -jar mutomvo.jar -p '"${1%.*}"' -g
# TotalTests=99
# TotalMutants=48
# StartingMutant=1

# [optimizations]
# DistributeOriginal=0
# SortTestSuite=0
# ScatterWorkload=0
# ClusterMutants=0
# ParallelCompilation=0
# ParallelMd5sum=0
# MultipleCoordinators=0

# [standalone]
# Standalone=1
# TestSuiteFile=/localStorage/mutomvo/project_'"${1%.*}"'/testsFile.txt

# [compilation]
# CompilationEnabled=1
# CompilationLineOriginal=gcc -O3 -lm -Wall [[ORIGINAL_PATH]]/'"${1%.*}"'.c -o [[ORIGINAL_PATH]]/'"${1%.*}"' 
# CompilationLineMutants=gcc -O3 -lm -Wall [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/'"${1%.*}"'.c -o [[MUTANTS_PATH]]/[[INDEX_MUTANT]]/'"${1%.*}"'
# CompilationNumWorkers=3
# CompilationWithScript=0
# CompilationScript=

# [timeouts]
# MALONE_MAX_ORIGINAL_TIMEOUT=90
# MALONE_MAX_MUTANTS_TIMEOUT_FACTOR=5
# MALONE_MAX_MUTANTS_MINIMUM_TIME=17

# [monitor]
# MonitorEnabled=0
# MonitorLines=vmstat,ip -s link,top -n 1 -b
# MonitorOnceLines=sysctl -a,lscpu,cat /proc/meminfo
# MonitorFrequency=60

# [misc]
# MarkerToken=
# MutantGenerationEnabled=0
# ' > "${1%.*}"_stand.ini

########################### QUE CAMBIAR ###########################

# /localStorage/mutomvo -> $MUTOMVO_HOME (no funciona)

# 1. 2
# 2. 3
# 3. 4
# 4. 8
# 5. 24

# tfgheras -> "${1%.*}"

# 1. 4
# 2. 5
# 3. 8
# 4. 24
# 5. 28
# 6. 28
# 7. 29
# 8. 29