// A recursive C program for partition problem
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// A utility function that returns true if there is
// a subset of arr[] with sum equal to given sum
bool isSubsetSum(int arr[], int n, int sum)
{
	// Base Cases
	if (sum == 0)
		return true;
	if (n == 0 && sum != 0)
		return false;

	// If last element is greater than sum, then
	// ignore it
	if (arr[n - 1] > sum)
		return isSubsetSum(arr, n - 1, sum);

	/* else, check if sum can be obtained by any of
	the following
	(a) including the last element
	(b) excluding the last element
	*/
	return isSubsetSum(arr, n - 1, sum)
		|| isSubsetSum(arr, n - 1, sum - arr[n - 1]);
}

// Returns true if arr[] can be partitioned in two
// subsets of equal sum, otherwise false
bool findPartiion(int arr[], int n)
{
	// Calculate sum of the elements in array
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += arr[i];

	// If sum is odd, there cannot be two subsets
	// with equal sum
	if (sum % 2 != 0)
		return false;

	// Find if there is subset with sum equal to
	// half of total sum
	return isSubsetSum(arr, n, sum / 2);
}

/* Driver program to test above function */
int main(int argc, char const *argv[]) { 
    // int array[] = {89, 32, 20, 113, -15}; 
    int array[argc];
    for (int i = 0; i < argc-1; i++){
        array[i] = atoi(argv[i+1]);
    }
    int n = argc-1;

	// Function call
	if (findPartiion(array, n) == true)
		printf("Can be divided into two subsets "
			"of equal sum\n");
	else
		printf("Can not be divided into two subsets"
			" of equal sum\n");
	return 0;
}
