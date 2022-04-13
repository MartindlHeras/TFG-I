#include <stdio.h> 
#include <stdlib.h>

// Insertion Sort Function
void insertionSort(int array[], int n) { 
    int i, element, j; 
    for (i = 1; i < n; i++) { 
        element = array[i]; j = i - 1; 
        while (j >= 0 && array[j] > element) { 
            array[j + 1] = array[j]; 
            j = j - 1; 
        } 
        array[j + 1] = element; 
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
    insertionSort(array, size); 
    printf("Sorted array: \n"); 
    printArray(array, size); 
    return 0; 
}