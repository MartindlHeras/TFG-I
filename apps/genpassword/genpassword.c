// genpassword.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef GENPASSWORD
#define GENPASSWORD

	#define ARR_SIZE(arr) ( sizeof((arr)) / sizeof((arr[0])) )

	// Declare functions
	char *generate_password(int length);

#endif

char *generate_password(int length) {
	char *pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	char *password = malloc( sizeof(*password) * (length + 1) );

	int i;
	for (i = 0; i < length; i++) {
		password[i] = pool[rand() % ARR_SIZE(pool)];
	}
	password[length] = '\0';

	return password;
}
// Testing for genpassword.c


#define true 1
#define false 0

// Note: since we use rand(), these will always return the same values.
int main(int argc, char * argv[]) 
{
	int i, sum = 0;
	if (argc != 2) {
		printf("You have forgot to specify two numbers.");
		exit(1);
	}
	int passed = true;

	// Seed
	srand(1337);

	// First test
	int length = atoi(argv[1]);
	char *password = generate_password(length);

	
	if (password != NULL) {
		printf("Password: %s\n",password);
	} 
	else
	    passed = false;

	// free memory
	free(password);


	return !passed;
}
