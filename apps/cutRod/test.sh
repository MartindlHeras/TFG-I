rm partitionProblem.zip
mv partitionProblem.c cutRod.c
mv tests_partitionProblem.txt tests_cutRod.txt
gcc -Wall -O3 cutRod.c -o cutRod
./cutRod 93 45 72 13 49
./cutRod 10 -60 -99 36 77
./cutRod 88 33 22 36 77
./cutRod 85 27 -58 -60 16
./cutRod 89 53 98 76 13
./cutRod 86 19 57 56 85
./cutRod 47 81 100 90 79
./cutRod 6 56 67 44 48
./cutRod 3 85 56 13 80
./cutRod 100 19 4 63 99
./cutRod 45 8
./cutRod a b
./cutRod 88 5
./cutRod 83 80
./cutRod 70 4
./cutRod 4 11
./cutRod 53 79
./cutRod 11 58
./cutRod 49 9
./cutRod 64 56
./cutRod 71
./cutRod 9
./cutRod 34
./cutRod 20
./cutRod 50
./cutRod 79 66 56 -74 25 0.80 1 27 96 20 -54 37 55 -60 11 -100 -90 65 97 62
./cutRod 59 47 46 66 65 50 67 6 82 77 24 53 33 34 95 21 55 60 76 86
./cutRod 32 20 26 25 69 22 24 84 -91 95 90 -49 34 -96 -47 43 42 72 83 65
./cutRod 28 61 36 3 51 45 35 100 39 53 9 76 21 11 88 97 41 80 1 44
./cutRod 95 29 77 79 ---15 17 --12 39 65 57 +53 63 48 14 80 35 93 27 56 38
./cutRod 7 84 88 35 14 12 8 -71 78 11 91 25 54 2 83 92 22 81 65 48
./cutRod 5 67 20 60 23 76 27 48 30 11 36 24 64 62 34 37 88 19 98 8
./cutRod 92 41 15 57 71 63 82 62 73 72 31 77 39 60 76 50 87 90 66 45
./cutRod 63 64 22 33 35 43 37 73 74 3 48 29 54 11 98 86 56 55 92 57
./cutRod 60 95 93 85 67 87 37 77 80 11 61 38 83 88 52 12 10 41 24 91
./cutRod 42 33 66 19 36 71 72 4 74 52 70 88 51 57 78 84 12 16 23 35 65 22 54 49 24 77 96 28 56 31
./cutRod 77 12 20 31 53 55 94 80 79 17 21 0000076 47 1 44 52 39 85 68 60 11 49 18 82 58 81 2 100 78 3
./cutRod 54 26 21 46 38 31 40 23 86 33 94 20 11 76 59 69 42 30 18 34 45 44 16 51 81 89 60 84 80 43
./cutRod 76 31 28 3 79 62 51 93 35 78.5 9 91 16 12 88 8 11 36 4 64 7 2 87 74 84 19 67 52 38 95
./cutRod 30 4 41 68 60 38 18 57 90 j 70 80 11 48 22 12 45 1 83 37 84 6 8 71 5 66 61 74 50 33
./cutRod 53 91 99 37 3 34 30 11 17 76 45 88 44 84 7 27 82 93 13 68 59 73 87 55 46 49 19 48 15 77
./cutRod 11 68 4 48 91 94 99 26 1 40-5 39 66 33 30 90 50 55 45 51 58 97 15 7 86 79 46 8 81 96 71
./cutRod 67 16 92 13 39 54 -51 93 27 40 78 59- 34 96 79 73 3 35 76 84 44 42 66 71 31 30 57 17 74 37
./cutRod 33 97 35 10 86 68 61 40 14 3 39 34 46 100 73 64 11 70 85 77 60 25 43 93 31 59 4 57 29 50
./cutRod "27 68 96 43 38 88 77 62 -24 69 75 92 18 11 13 70 85 25 58 22 98 76 52 32 71 73 26 60 19 64"
./cutRod 17 58 20 60 95 94 37 44 48 24 5 10 30 6 63
./cutRod 41 69 90 6 -96 15 58 76 18 -42 -74 -80 -27 -95 79
./cutRod 50 99 77 30 -80 26 34 -6 18 -89 2 -75 1 37 14
./cutRod 37 96 69 70 -58 8 30 6 -45 33 53 83 31 71 47
./cutRod 85 88 52 79 30 37 66 61 5 7 25 16 53 44 58
rm cutRod
zip -r cutRod.zip cutRod.c