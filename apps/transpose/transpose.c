#include <stdio.h>
#include <stdlib.h>
int main(int argc, char* argv[]) {
    int a[10][10], transpose[10][10], r, c;

    // reading rows and columns
    r = atoi(argv[1]);
    c = atoi(argv[2]);

    // reading the matrix
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            a[i][j] = atoi(argv[3+i*c+j]);
        }

    // printing the matrix a[][]
    printf("\nEntered matrix: \n");
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            printf("%d  ", a[i][j]);
            if (j == c - 1)
            printf("\n");
        }

    // computing the transpose
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            transpose[j][i] = a[i][j];
        }

    // printing the transpose
    printf("\nTranspose of the matrix:\n");
    for (int i = 0; i < c; i++)
        for (int j = 0; j < r; j++) {
            printf("%d  ", transpose[i][j]);
            if (j == r - 1)
            printf("\n");
        }
    return 0;
}