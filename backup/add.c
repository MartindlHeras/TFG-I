#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[]) 
{
	int i, sum = 0;
	if (argc != 3) {
		printf("You have forgot to specify two numbers.");
		exit(1);
	}
	sum= atoi(argv[1])+atoi(argv[2]);
	printf("%d\n", sum);

}
