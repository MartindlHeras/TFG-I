// C code to linearly search x in arr[]. If x
// is present then return its location, otherwise
// return -1
 
#include <stdio.h>
#include <stdlib.h>
 
int search(int arr[], int n, int x)
{
    int i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}
 
// Driver code
int main(int argc, char const *argv[]) { 
    // int array[] = {89, 32, 20, 113, -15};
	if (argc < 2)
	{
		printf("Too few arguments\n");
		return -1;
	}
	
    int array[argc-1];
	int x = atoi(argv[1]);
    for (int i = 1; i < argc-1; i++){
        array[i] = atoi(argv[i+1]);
    }
    int size = argc-2;
    // int size = sizeof(array)/sizeof(array[0]);
   
    // Function call
    int result = search(array, size, x);
    (result == -1)
        ? printf("Element is not present in array\n")
        : printf("Element is present at index %d\n", result);
    return 0;
}