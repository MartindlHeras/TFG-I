#!/bin/bash
###################################################################
##
## lineas del .c OK
## num tests OK
## num mutantes OK
## tiempo total OK +-
## tiempo de cada mutante +-
## cores ?
## mutantes vivos/muertos (no creo porque eso nunca se sabe) ?
## CALCULAR EL MENOR TIEMPO Y QUE ALGORITMO LO HA HECHO Y SOLO METER ESOS DATOS
##
###################################################################

# grep -e 'Total elapsed time: ' output2.txt >> "${1%.*}"_output.txt

########### OBTENER EL MENOR TIEMPO Y GUARDAR EL ALGORITMO
########### PILLAR LO DEL 2 E IR COMPARANDO 1-1 CON LOS SUCESIVOS

# grep -e 'Total elapsed time: ' output2.txt >> "${1%.*}"_output.txt
# grep -e 'Total elapsed time: ' output3.txt >> "${1%.*}"_output.txt
# grep -e 'Total elapsed time: ' output4.txt >> "${1%.*}"_output.txt
grep 'Total elapsed time: ' output*.txt #>> "${1%.*}"_output.txt
grep -oE '[^ ]+$' -e 'Total elapsed time: ' output*.txt

########### UNA VEZ TENGA EL MAYOR, GUARDO ESE ALGORITMO Y EL TIEMPO Y BUSCO EL TIEMPO DE LOS MUTANTES?