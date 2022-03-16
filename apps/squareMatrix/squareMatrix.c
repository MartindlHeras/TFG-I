#include <stdio.h>

void enterData(int firstMatrix[][10], int secondMatrix[][10], int rowFirst, int columnFirst, int rowSecond, int columnSecond);
void multiplyMatrices(int firstMatrix[][10], int secondMatrix[][10], int multResult[][10], int rowFirst, int columnFirst, int rowSecond, int columnSecond);
void display(int mult[][10], int rowFirst, int columnSecond);

int main(int argc, char const *argv[])
{
	int firstMatrix[10][10], secondMatrix[10][10], mult[10][10], rowFirst, columnFirst, rowSecond, columnSecond, i, j, k;

	// reading rows and columns
    rowFirst = atoi(argv[1]);
    columnFirst = atoi(argv[2]);
	rowSecond = atoi(argv[1]);
    columnSecond = atoi(argv[2]);

    // reading the matrix
    for (int i = 0; i < rowFirst; i++)
        for (int j = 0; j < columnFirst; j++) {
            firstMatrix[i][j] = atoi(argv[3+i*columnFirst+j]);
			secondMatrix[i][j] = atoi(argv[3+i*columnFirst+j]);
        }

	// Function to multiply two matrices.
	multiplyMatrices(firstMatrix, secondMatrix, mult, rowFirst, columnFirst, rowSecond, columnSecond);

	// Function to display resultant matrix after multiplication.
	display(mult, rowFirst, columnSecond);

	return 0;
}

void multiplyMatrices(int firstMatrix[][10], int secondMatrix[][10], int mult[][10], int rowFirst, int columnFirst, int rowSecond, int columnSecond)
{
	int i, j, k;

	// Initializing elements of matrix mult to 0.
	for(i = 0; i < rowFirst; ++i)
	{
		for(j = 0; j < columnSecond; ++j)
		{
			mult[i][j] = 0;
		}
	}

	// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
	for(i = 0; i < rowFirst; ++i)
	{
		for(j = 0; j < columnSecond; ++j)
		{
			for(k=0; k<columnFirst; ++k)
			{
				mult[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
			}
		}
	}
}

void display(int mult[][10], int rowFirst, int columnSecond)
{
	int i, j;
	printf("\nOutput Matrix:\n");
	for(i = 0; i < rowFirst; ++i)
	{
		for(j = 0; j < columnSecond; ++j)
		{
			printf("%d  ", mult[i][j]);
			if(j == columnSecond - 1)
				printf("\n\n");
		}
	}
}