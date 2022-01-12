#include <stdio.h>
int main(int argc, char * argv[]) {
  int n, i, flag = 0;
  n = atoi(argv[1]);

  for (i = 2; i <= n / 2; ++i) {

    // if n is divisible by i, then n is not prime
    // change flag to 1 for non-prime number
    if (n % i == 0) {
      flag = 1;
      break;
    }
  }

  // 0 and 1 are not prime numbers
  if (n == 0 || n == 1) {
    printf("%d is neither prime nor composite.", n);
  } 
  else {

    // flag is 0 for prime numbers
    if (flag == 0)
      printf("%d is a prime number.", n);
    else
      printf("%d is not a prime number.", n);
  }

  return 0;
}