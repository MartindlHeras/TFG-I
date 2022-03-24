int main(int argc, char **argv)
{  
  double j=0;
  double i=0;

  if(argc == 2)
  {
    j = atoi(argv[1]);
    for (i = 0; i < j*j*j*j*j; i++)  {}
  }

  printf("%f\n", i);
}
