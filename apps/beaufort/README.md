## beaufort.c
 - overview: A C implementation of the Beaufort Cipher
 - input: flags and a pipe (see the example)
 - output: 
     - print: message 
     - return: 0
 - #mutants: 481
 - #lines: 445
 - test format: beaufort usage: beaufort [-hV] [options]

    options:

    --encrypt           encrypt stdin stream
    --decrypt           decrypt stdin stream
    --key=[key]         cipher key (required)
    --alphabet=[alpha]  cipher tableau alphabet (Default: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    --text=[text]       text to either encrypt or decrypt


echo 'ay mi madre' | ./beaufort --encrypt --key=panda
echo echo 'Fc 1v oFxwz' | ./beaufort --decrypt --key=panda
