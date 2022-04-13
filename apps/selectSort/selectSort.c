#include <stdio.h> 
#include <stdlib.h>
// Function to swap elements 
void swap(int *a, int *b) { 
    int temp = *a; 
    *a = *b; 
    *b = temp; 
}  

// Selection Sort
void selectionSort(int array[], int n) { 
    int i, j, min_element; 
    for (i = 0; i < n-1; i++) {
        min_element = i; 
        for (j = i+1; j < n; j++) 
            if (array[j] < array[min_element]) 
                min_element = j; 
        swap(&array[min_element], &array[i]); 
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
    selectionSort(array, size); 
    printf("Sorted array: \n"); 
    printArray(array, size); 
    return 0; 
}