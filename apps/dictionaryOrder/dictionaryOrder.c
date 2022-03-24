#include <stdio.h>
#include <string.h>

int main(int argc, char const *argv[]){
   char str[5][50], temp[50];

   // Getting strings input
   for (int i = 0; i < 5; ++i) {
      strcpy(str[i], argv[i+1]);
   }

   // storing strings in the lexicographical order
   for (int i = 0; i < 5; ++i) {
      for (int j = i + 1; j < 5; ++j) {

         // swapping strings if they are not in the lexicographical order
         if (strcmp(str[i], str[j]) > 0) {
            strcpy(temp, str[i]);
            strcpy(str[i], str[j]);
            strcpy(str[j], temp);
         }
      }
   }

   printf("\nIn the lexicographical order: \n");
   for (int i = 0; i < 5; ++i) {
      // fputs(str[i], stdout);
      printf("%s ", str[i]);
   }
   printf("\n");
   return 0;
}