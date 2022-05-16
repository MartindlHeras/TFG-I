rm fibonacciSearch.zip
mv fibonacciSearch.c exponentialSearch.c
mv tests_fibonacciSearch.txt tests_exponentialSearch.txt
gcc -Wall -O3 exponentialSearch.c -o exponentialSearch
./exponentialSearch 13 13 45 49 72 93
./exponentialSearch 77 10 36 60 77 99
./exponentialSearch 23 22 33 36 77 88
./exponentialSearch a 16 27 58 60 85
./exponentialSearch 76 13 53 76 89 98
./exponentialSearch 86 19 56 57 85
./exponentialSearch 47 47 79 81 90 100
./exponentialSearch 6 6 44 48 56 67
./exponentialSearch 3 13 56 80 85
./exponentialSearch 100 4 19 63 99
./exponentialSearch 45 8 45
./exponentialSearch 9 58
./exponentialSearch 88 5 5 5 5 5 5 5
./exponentialSearch 83 80 83 83 83 83 83
./exponentialSearch 70 4 70
./exponentialSearch 4 11
./exponentialSearch 53 79
./exponentialSearch 11 11
./exponentialSearch 49 9
./exponentialSearch 64 56
./exponentialSearch 71 
./exponentialSearch 9 
./exponentialSearch 34 0000034
./exponentialSearch 20 21
./exponentialSearch 50 "50"
./exponentialSearch 79 1 11 20 a 25 27 37 54 55 56 60 62 65 66 74 79 80 90 96 97 100
./exponentialSearch 59 6 21 24 33 34 46 47 50 53 55 60 65 66 67 76 77 82 86 95
./exponentialSearch 32 4 9 20 22 24 25 26 32.0000 34 42 43 47 65 69 72 83 84 90 91 95 96
./exponentialSearch 28 1 3 9 11 21 35 36 39 41 44 45 51 53 61 76 80 88 97 100
./exponentialSearch 95 0.12 14 15 17 27 29 35 38 39 48 53 56 57 63 65 77 79 80 93 95
./exponentialSearch 7 2 8 11 12 14 22 25 35 48 54 65 71 78 81 83 84 88 91 92
./exponentialSearch 5 8 11 19 20 23 24 27 30 34 36 37 48 60 62 64 67 76 88 98
./exponentialSearch 92 15 31 39 41 45 50 57 60 62 63 66 71 72 73 76 77 82 87 90
./exponentialSearch 63 3 11 22 29 33 35 37 43 48 54 55 56 57 64 73 74 86 92 98
./exponentialSearch 60 10 11 12 24 37 38 41 52 60 61 67 77 80 83 85 87 88 91 93 95
./exponentialSearch 42 4 4 7 12 16 19 22 23 24 28 31.5 33 35 36 42 49 51 52 54 56 57 65 66 70 71 72 77 78 84 88 96
./exponentialSearch 77 1 2 3 11 12 17 18 20 21 31 39 44 47 49 52 53 55 58 60 68 76 78 79 80 81 82 85 94 100
./exponentialSearch 54 11 16 18 20 21 23 26 30 31 33 34 38 40 42 43 44 45 46 51 59 60 69 76 80 81 84 86 89 94
./exponentialSearch 76 2 3 4 7 8 9 11 12 16 19 28 31 35 36 38 51 52 62 64 67 74 78 79 84 87 88 91 93 95
./exponentialSearch 30 0 1 4 5 6 7 8 11 12 18 22 30 33 37 38 41 45 48 50 57 60 61 66 68 71 74 78 80 83 84 90
./exponentialSearch 53 3 7 11 13 15 17 19 27 30 34 37 44 45 46 48 49 55 59 68 73 76 77 82 84 87 88 91 93 99
./exponentialSearch 11 1 4 7 8 15 26 30 33 39 40 45 46 48 50 51 55 58 66 68 71 79 81 86 90 91 94 96 97 99
./exponentialSearch 67 3 5 9 13 16 17 27 30 31 34 35 37 39 40 42 44 51 54 57 66 67 71 73 74 76 78 79 84 92 93 96
./exponentialSearch 33 "3 4 10 11 14 25 29 31 34 35 39 40 43 46 50 57 59 60 61 64 68 70 73 77 85 86 93 97 100"
./exponentialSearch 27 0 0 0 0 0 0 0 0 0 0 0 0 0 0
./exponentialSearch 17 5 6 10 17 20 24 30 37 44 48 58 60 63 94 95
./exponentialSearch 41 6 15 18 27 42 58 69 74 76 79 80 90 95 96
./exponentialSearch 50 1 1 2 6 14 18 26 30 34 37 1/77 50 75 80 89 99
./exponentialSearch 37 6 8 30 31 33 45 47 53 58 69 70 71 83 96
./exponentialSearch 85 5 7 16 25 30 37 44 52 53 58 61 66 79 85 88
rm exponentialSearch
zip -r exponentialSearch.zip exponentialSearch.c