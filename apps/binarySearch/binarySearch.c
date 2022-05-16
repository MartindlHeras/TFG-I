// C program to implement recursive Binary Search
#include <stdio.h>
#include <stdlib.h>

// A recursive binary search function. It returns
// location of x in given array arr[l..r] is present,
// otherwise -1
int binarySearch(int arr[], int l, int r, int x)
{
	if (r >= l) {
		int mid = l + (r - l) / 2;

		// If the element is present at the middle
		// itself
		if (arr[mid] == x)
			return mid;

		// If element is smaller than mid, then
		// it can only be present in left subarray
		if (arr[mid] > x)
			return binarySearch(arr, l, mid - 1, x);

		// Else the element can only be present
		// in right subarray
		return binarySearch(arr, mid + 1, r, x);
	}

	// We reach here when element is not
	// present in array
	return -1;
}

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
	int result = binarySearch(arr, 0, n - 1, x);
	(result == -1)
		? printf("Element is not present in array\n")
		: printf("Element is present at index %d\n", result);
	return 0;
}
