#include <stdio.h> 
#include <stdlib.h>
// Function to swap elements 
void swap(int *a, int *b) { 
    int temp = *a; 
    *a = *b; 
    *b = temp; 
}  

// bubble sort function
void bubbleSort(int array[], int n) { 
    int i, j; 
    for (i = 0; i < n-1; i++)       
    for (j = 0; j < n-i-1; j++) if (array[j] > array[j+1]) 
    swap(&array[j], &array[j+1]); 
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
    bubbleSort(array, size); 
    printf("Sorted array: \n"); 
    printArray(array, size); 
    return 0; 
}