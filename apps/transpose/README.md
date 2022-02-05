## transpose.c

Falla porque en los bucles for, al eliminar el ++ de i++/j++ crea bucles infinitos

 - overview: C program to transpose a matrix
 - input: dimensions of the matrix and the matrix
 - output: 
     - print: printed matrix and printed transposed matrix
     - return: 0
 - #mutants: 148
 - #lines: 39
 - test format: transpose \<rows\> \<columns\> \<matrix\>