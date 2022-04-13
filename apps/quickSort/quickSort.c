#include <stdio.h> 
#include <stdlib.h>
// Function to swap elements 
void swap(int *a, int *b) { 
    int temp = *a; 
    *a = *b; 
    *b = temp; 
}  

// Partition function
int partition (int arr[], int lowIndex, int highIndex) { 
    int pivotElement = arr[highIndex];
    int i = (lowIndex - 1); 
    for (int j = lowIndex; j <= highIndex- 1; j++) { 
        if (arr[j] <= pivotElement) { 
            i++; 
            swap(&arr[i], &arr[j]); 
        } 
    } 
    swap(&arr[i + 1], &arr[highIndex]); 
    return (i + 1); 
}   
// QuickSort Function
void quickSort(int arr[], int lowIndex, int highIndex) { 
    if (lowIndex < highIndex) { 
        int pivot = partition(arr, lowIndex, highIndex); 
        // Separately sort elements before & after partition 
        quickSort(arr, lowIndex, pivot - 1); 
        quickSort(arr, pivot + 1, highIndex); 
    } 
}  

// Function to print the elements of an array
void printArray(int array[], int size) { 
    int i; 
    for (i=0; i < size; i++) 
    printf("%d ", array[i]); 
    printf("\n"); 
}   

// Main Function
int main(int argc, char const *argv[]) { 
    // int array[] = {89, 32, 20, 113, -15}; 
    int array[argc];
    for (int i = 0; i < argc-1; i++){
        array[i] = atoi(argv[i+1]);
    }
    int size = argc-1;
    // int size = sizeof(array)/sizeof(array[0]);
    printf("Given array: \n");
    printArray(array, size);
    quickSort(array, 0, size-1); 
    printf("Sorted array: \n"); 
    printArray(array, size); 
    return 0; 
}