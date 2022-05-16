// CPP program to find maximum product of
// a subset.
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

// Find maximum between two numbers.
int max(int num1, int num2)
{
	return (num1 > num2) ? num1 : num2;
}

// Find minimum between two numbers.
int min(int num1, int num2)
{
	return (num1 > num2) ? num2 : num1;
}

int minProductSubset(int a[], int n)
{
	if (n == 1)
		return a[0];
	// Find count of negative numbers, count of zeros,
	// maximum valued negative number, minimum valued
	// positive number and product of non-zero numbers
	int max_neg = INT_MIN, min_pos = INT_MAX, count_neg = 0,
		count_zero = 0, prod = 1;
	for (int i = 0; i < n; i++) {
		// If number is 0, we don't multiply it with
		// product.
		if (a[i] == 0) {
			count_zero++;
			continue;
		}
		// Count negatives and keep track of maximum valued
		// negative.
		if (a[i] < 0) {
			count_neg++;
			max_neg = max(max_neg, a[i]);
		}
		// Track minimum positive number of array
		if (a[i] > 0)
			min_pos = min(min_pos, a[i]);
		prod = prod * a[i];
	}
	// If there are all zeros or no negative number present
	if (count_zero == n || (count_neg == 0 && count_zero > 0))
		return 0;
	// If there are all positive
	if (count_neg == 0)
		return min_pos;
	// If there are even number of negative numbers and
	// count_neg not 0
	if (!(count_neg & 1) && count_neg != 0)
		// Otherwise result is product of all non-zeros
		// divided by maximum valued negative.
		prod = prod / max_neg;
	return prod;
}

int main(int argc, char const *argv[]) { 
    // int array[] = {89, 32, 20, 113, -15}; 
    int array[argc];
    for (int i = 0; i < argc-1; i++){
        array[i] = atoi(argv[i+1]);
    }
    int n = argc-1;
	printf("Minimum product subset of array: %d\n", minProductSubset(array, n));
	return 0;
}

// This code is contributed by Sania Kumari Gupta
