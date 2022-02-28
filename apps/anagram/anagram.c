#include<stdio.h>
#include<string.h>

int main(int argc, char const *argv[])
{
    int len, len1, len2, i, j, found=0, not_found=0;
    len1 = strlen(argv[1]);
    len2 = strlen(argv[2]);
    if(len1 == len2)
    {
        len = len1;
        for(i=0; i<len; i++)
        {
            found = 0;
            for(j=0; j<len; j++)
            {
                if(argv[1][i] == argv[2][j])
                {
                    found = 1;
                    break;
                }
            }
            if(found == 0)
            {
                not_found = 1;
                break;
            }
        }
        if(not_found == 1)
            printf("Strings are not Anagram\n");
        else
            printf("Strings are Anagram\n");
    }
    else
        printf("Strings are not Anagram\n");
    return 0;
}