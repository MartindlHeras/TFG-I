
/**
 * `encrypt.c' - libbeaufort
 *
 * copyright (c) 2014 joseph werle <joseph.werle@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>


/**
 * `beaufort.h' - libbeaufort
 *
 * copyright (c) 2014 joseph werle <joseph.werle@gmail.com>
 */

#ifndef BEAUFORT_H
#define BEAUFORT_H 1

#if __GNUC__ >= 4
# define BEAUFORT_EXTERN __attribute__((visibility("default")))
#else
# define BEAUFORT_EXTERN
#endif

/**
 * Default beaufort alphabet
 */

#ifndef BEAUFORT_ALPHA
#define BEAUFORT_ALPHA \
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#endif

/**
 * beaufort version
 */

#ifndef BEAUFORT_VERSION
#define BEAUFORT_VERSION "1" // default
#endif

/**
 * Encrypts plaintext using the beaufort
 * cipher and an alphabet table if provided.
 * The function will fallback to a table
 * generated using the default alphabet
 * defined by `BEAUFORT_ALPHA`.
 */

BEAUFORT_EXTERN char *
beaufort_encrypt (const char *, const char *, char **);

/**
 * Decrypts plaintext using the beaufort
 * cipher and an alphabet table if provided.
 * The function will fallback to a table
 * generated using the default alphabet
 * defined by `BEAUFORT_ALPHA`.
 */

BEAUFORT_EXTERN char *
beaufort_decrypt (const char *, const char *, char **);

/**
 * Constructs a tableau of rows
 * and columns from a given alphabet
 * using the following forumula to determine
 * the row index in each column during creation:
 * `(j + y) % size` where `size = strlen(alpha)`,
 * `j = size - 1`, and * `y = y + 1`
 */

BEAUFORT_EXTERN char **
beaufort_tableau (const char *);

#endif


/**
 * `decrypt.c' - libbeaufort
 *
 * copyright (c) 2014 joseph werle <joseph.werle@gmail.com>
 */

static size_t
ssize (const char *str) {
  size_t size = 0;
  while ('\0' != str[size]) size++;
  return size;
}

char *
beaufort_decrypt (const char *src, const char *key, char **mat) {
  char *dec = NULL;
  char ch = 0;
  char k = 0;
  size_t ksize = 0;
  size_t size = 0;
  size_t rsize = 0;
  size_t len = 0;
  int i = 0;
  int x = 0;
  int y = 0;
  int j = 0;
  int needed = 1;

  if (NULL == mat) {
    mat = beaufort_tableau(BEAUFORT_ALPHA);
    if (NULL == mat) { return NULL; }
  }

  ksize = ssize(key);
  len = ssize(src);
  rsize = ssize(mat[0]);
  dec = (char *) malloc(sizeof(char) * len + 1);

  if (NULL == dec) { return NULL; }

  for (; (ch = src[i]); ++i) {
    needed = 1;

    // find column with char
    for (y = 0; y < rsize; ++y) {
      if (ch == mat[y][0]) { needed = 1; break; }
      else { needed = 0; }
    }

    // if not needed append
    // char and continue
    if (0 == needed) {
      dec[size++] = ch;
      continue;
    }

    // determine char in `key'
    k = key[(j++) % ksize];

    for (x = 0; x < rsize; ++x)  {
      if (k == mat[y][x]) { needed = 1; break; }
      else { needed = 0; }
    }

    // append current char if not
    // needed and decrement unused
    // modulo index
    if (0 == needed) {
      dec[size++] = ch;
      j--;
      continue;
    }

    dec[size++] = mat[0][x];
  }

  dec[size] = '\0';

  return dec;
}

/*encrypt*/



char *
beaufort_encrypt (const char *src, const char *key, char **mat) {
  char *enc = NULL;
  char ch = 0;
  char k = 0;
  size_t ksize = 0;
  size_t size = 0;
  size_t len = 0;
  size_t rsize = 0;
  int i = 0;
  int x = 0;
  int y = 0;
  int j = 0;
  int needed = 1;

  if (NULL == mat) {
    mat = beaufort_tableau(BEAUFORT_ALPHA);
    if (NULL == mat) { return NULL; }
  }

  ksize = ssize(key);
  len = ssize(src);
  rsize = ssize(mat[0]);
  enc = (char *) malloc(sizeof(char) * len + 1);

  if (NULL == enc) { return NULL; }

  for (; (ch = src[i]); ++i) {
    // reset
    needed = 1;

    // column with `ch' at top
    for (x = 0, y = 0; x < rsize; ++x) {
      if (ch == mat[y][x]) { needed = 1; break; }
      else { needed = 0; }
    }

    // if char not in top row
    // append current char
    if (0 == needed) {
      enc[size++] = ch;
      continue;
    }

    // determine char in `key'
    k = key[(j++) % ksize];

    // find row in column with `key[k]'
    for (y = 0; y < rsize; ++y) {
      if (k == mat[y][x]) { needed = 1; break; }
      else { needed = 0; }
    }

    // append char and decrement
    // unused modolu index if
    // not needed
    if (0 == needed) {
      enc[size++] = ch;
      j--;
      continue;
    }

    // append left char
    enc[size++] = mat[y][0];
  }

  enc[size] = '\0';

  return enc;
}

/**
 * `tableau.c' - libbeaufort
 *
 * copyright (c) 2014 joseph werle <joseph.werle@gmail.com>
 */

char **
beaufort_tableau (const char *alpha) {
  size_t size = ssize(alpha);
  char **mat = NULL;
  int x = 0;
  int y = 0;
  int j = 0;

  mat = (char **) calloc(size + 1, sizeof(char *));

  if (NULL == mat) { return NULL; }

  for (;y < size; ++y) {
    mat[y] = (char *) calloc(size, sizeof(char));

    if (NULL == mat[y]) { return NULL; }

    for (x = 0, j = size; x < size; ++x, --j) {
      mat[y][x] = alpha[(j + y) % size];
    }

    mat[y][x] = '\0';
  }

  mat[y] = NULL;

  return mat;
}
/**
 * `main.c' - libbeaufort
 *
 * copyright (c) 2014 joseph werle <joseph.werle@gmail.com>
 */

enum { NO_OP, ENCRYPT_OP, DECRYPT_OP };

static void
usage () {
  fprintf(stderr, "usage: beaufort [-hV] [options]\n");
}

static void
help () {
  fprintf(stderr, "\noptions:\n");
  fprintf(stderr, "\n  --encrypt           encrypt stdin stream");
  fprintf(stderr, "\n  --decrypt           decrypt stdin stream");
  fprintf(stderr, "\n  --key=[key]         cipher key (required)");
  fprintf(stderr,
      "\n  --alphabet=[alpha]  cipher tableau alphabet (Default: '%s')\n",
      BEAUFORT_ALPHA);
  fprintf(stderr, "\n");
}

static char *
read_stdin () {
  size_t bsize = 1024;
  size_t size = 1;
  char buf[bsize];
  char *res = (char *) malloc(sizeof(char) * bsize);
  char *tmp = NULL;

  // memory issue
  if (NULL == res) { return NULL; }

  // cap
  res[0] = '\0';

  // read
  if (NULL != fgets(buf, bsize, stdin)) {
    // store
    tmp = res;
    // resize
    size += (size_t) strlen(buf);
    // realloc
    res = (char *) realloc(res, size);

    // memory issues
    if (NULL == res) {
      free(tmp);
      return NULL;
    }

    // yield
    strcat(res, buf);

    return res;
  }

  free(res);

  return NULL;
}

int
main (int argc, char **argv) {
  char *buf = NULL;
  char *alpha = NULL;
  char *key = NULL;
  char *out = NULL;
  char **mat = NULL;
  int op = NO_OP;

  // emit usage with empty arguments
  if (1 == argc) { return usage(), 1; }

  // parse opts
  {
    int i = 0;
    char *opt = NULL;
    char tmp = 0;

    opt = *argv++; // unused

    while ((opt = *argv++)) {

      // flags
      if ('-' == *opt++) {
        switch (*opt++) {
          case 'h':
            return usage(), help(), 0;

          case 'V':
            fprintf(stderr, "%s\n", BEAUFORT_VERSION);
            return 0;

          case '-':
            if (0 == strcmp(opt, "encrypt")) { op = ENCRYPT_OP;}
            if (0 == strcmp(opt, "decrypt")) { op = DECRYPT_OP;}

            // parse key
            if (0 == strncmp(opt, "key=", 4)) {
              for (i = 0; i < 4; ++i) tmp = *opt++;
              key = opt;
            }

            if (0 == strncmp(opt, "alphabet=", 8)) {
              for (i = 0; i < 9; ++i) tmp = *opt++;
              alpha = opt;
            }
            break;

          default:
            tmp = *opt--;
            // error
            fprintf(stderr, "unknown option: `%s'\n", opt);
            usage();
            return 1;
        }
      }
    }
  }

  if (NULL == alpha) {
    alpha = BEAUFORT_ALPHA;
  }

  mat = beaufort_tableau(alpha);

  if (NULL == key) {
    fprintf(stderr, "error: Expecting cipher key\n");
    usage();
    return 1;
  }

#define OP(name) {                               \
  buf = read_stdin();                            \
  if (NULL == buf) { return 1; }                 \
  out = beaufort_ ## name(buf, key, mat);        \
  printf("%s\n", out);                           \
  do {                                           \
    buf = read_stdin();                          \
    if (NULL == buf) { break; }                  \
    out = beaufort_ ## name(buf, key, mat);      \
    printf("%s\n", out);                         \
  } while (NULL != buf);                         \
}

switch (op) {
  case ENCRYPT_OP:
    if (1 == isatty(0)) { return 1; }
    else if (ferror(stdin)) { return 1; }
    else { OP(encrypt); }

    printf("Encrypt\n");
    return 0;

  case DECRYPT_OP:
    if (1 == isatty(0)) { return 1; }
    else if (ferror(stdin)) { return 1; }
    else { OP(decrypt); }
    return 0;

  case NO_OP:
  default:
    return usage(), 1;
}

#undef OP
return 0;
}
