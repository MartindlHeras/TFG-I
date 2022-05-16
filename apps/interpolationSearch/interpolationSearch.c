// C program to implement interpolation search
// with recursion
#include <stdio.h>
#include <stdlib.h>

// If x is present in arr[0..n-1], then returns
// index of it, else returns -1.
int interpolationSearch(int arr[], int lo, int hi, int x)
{
	int pos;
	// Since array is sorted, an element present
	// in array must be in range defined by corner
	if (lo <= hi && x >= arr[lo] && x <= arr[hi]) {
		// Probing the position with keeping
		// uniform distribution in mind.
		pos = lo
			+ (((double)(hi - lo) / (arr[hi] - arr[lo]))
				* (x - arr[lo]));

		// Condition of target found
		if (arr[pos] == x)
			return pos;

		// If x is larger, x is in right sub array
		if (arr[pos] < x)
			return interpolationSearch(arr, pos + 1, hi, x);

		// If x is smaller, x is in left sub array
		if (arr[pos] > x)
			return interpolationSearch(arr, lo, pos - 1, x);
	}
	return -1;
}

// Driver Code
int main(int argc, char const *argv[]) { 
    // int array[] = {89, 32, 20, 113, -15};
	if (argc < 2)
	{
		printf("Too few arguments\n");
		return -1;
	}
	
    int arr[argc-1];
	int x = atoi(argv[1]);
    for (int i = 1; i < argc-1; i++){
        arr[i] = atoi(argv[i+1]);
    }
    int n = argc-2;
    // int size = sizeof(array)/sizeof(array[0]);
	int index = interpolationSearch(arr, 0, n - 1, x);

	// If element was found
	if (index != -1)
		printf("Element found at index %d\n", index);
	else
		printf("Element not found.\n");
	return 0;
}