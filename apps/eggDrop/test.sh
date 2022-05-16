rm cutRod.zip
mv cutRod.c eggDrop.c
mv tests_cutRod.txt tests_eggDrop.txt
gcc -Wall -O3 eggDrop.c -o eggDrop
./eggDrop 5 4
./eggDrop 3 10
./eggDrop 10 5
./eggDrop 1 10
./eggDrop 4 8
./eggDrop 5 9
./eggDrop 9 10
./eggDrop 8 1
./eggDrop 9 8
./eggDrop 5
./eggDrop 9 6
./eggDrop 9 10
./eggDrop 4 -2
./eggDrop 7 3
./eggDrop a 1
./eggDrop 4 10
./eggDrop 10 7
./eggDrop 8 2
./eggDrop 5 4.5
./eggDrop 10 3
./eggDrop 7 9
./eggDrop 6 6
./eggDrop 7 1
./eggDrop 2 10
./eggDrop 9 9
./eggDrop 9 9
./eggDrop 9 7
./eggDrop 2 6
./eggDrop 6 8
./eggDrop 9 1
./eggDrop 10 7
./eggDrop 3 2
./eggDrop 3 1
./eggDrop 3 3
./eggDrop 5 6
./eggDrop 1 2
./eggDrop 7 4
./eggDrop 5 9
./eggDrop 2 6
./eggDrop 7 8
./eggDrop 7 9
./eggDrop 6 7
./eggDrop 5 7
./eggDrop 9 6
./eggDrop 4 9
./eggDrop 10 7
./eggDrop 6 9
./eggDrop 1 6
./eggDrop 9 10
./eggDrop 2 6
rm eggDrop
zip -r eggDrop.zip eggDrop.c