#include <stdio.h>
int main(int argc, char const *argv[])
{
    int n1, n2, i, gcd;

    n1 = atoi(argv[1]);
    n2 = atoi(argv[2]);

    for(i=1; i <= n1 && i <= n2; ++i)
    {
        // Checks if i is factor of both integers
        if(n1%i==0 && n2%i==0)
            gcd = i;
    }

    printf("G.C.D of %d and %d is %d", n1, n2, gcd);

    return 0;
}