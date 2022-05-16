// A Dynamic Programming solution for Rod cutting problem
#include<stdio.h>
#include<stdlib.h>
#include<limits.h>

// A utility function to get the maximum of two integers
int max(int a, int b) { return (a > b)? a : b;}

/* Returns the best obtainable price for a rod of length n and
price[] as prices of different pieces */
int cutRod(int price[], int n)
{
int val[n+1];
val[0] = 0;
int i, j;

// Build the table val[] in bottom up manner and return the last entry
// from the table
for (i = 1; i<=n; i++)
{
	int max_val = INT_MIN;
	for (j = 0; j < i; j++)
		max_val = max(max_val, price[j] + val[i-j-1]);
	val[i] = max_val;
}

return val[n];
}

/* Driver program to test above function */
int main(int argc, char const *argv[]) { 
    // int array[] = {89, 32, 20, 113, -15}; 
    int array[argc];
    for (int i = 0; i < argc-1; i++){
        array[i] = atoi(argv[i+1]);
    }
    int n = argc-1;
	// int size = sizeof(arr)/sizeof(arr[0]);
	printf("Maximum Obtainable Value is %d\n", cutRod(array, n));
	// getchar();
	return 0;
}
