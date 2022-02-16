## add.c
 - overview: A C implementation of the Beaufort Cipher
 - input: flags and a pipe (see the example)
 - output: 
     - print: message 
     - return: 0
 - #mutants: 11
 - #lines: 13
 - test format:  ??


echo 'ay mi madre' | ./tests --encrypt --key=panda
echo echo 'Fc 1v oFxwz' | ./tests --decrypt --key=panda
