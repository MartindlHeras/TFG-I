// C Program to count number of
// ways to reach Nth stair
#include <stdio.h>
#include <stdlib.h>

// A simple recursive program to
// find n'th fibonacci number
int fib(int n)
{
	if (n <= 1)
		return n;
	return fib(n - 1) + fib(n - 2);
}

// Returns number of ways to reach s'th stair
int countWays(int s)
{
	return fib(s + 1);
}

// Driver program to test above functions
int main(int argc, char const *argv[])
{
	if (argc < 2)
	{
		printf("Something went wrong.\n");
		return -1;
	}
	
	int s = atoi(argv[1]);
	printf("Number of ways = %d\n", countWays(s));
	// getchar();
	return 0;
}
