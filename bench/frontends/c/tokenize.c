#include <stddef.h>
#include <string.h>

enum token_kind { token_word, token_number, token_symbol, token_space };

struct counts {
  unsigned kinds[4];
  unsigned long digits;
  unsigned hash;
};

static enum token_kind classify(unsigned char byte) {
  if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
      byte == '_')
    return token_word;
  if (byte >= '0' && byte <= '9')
    return token_number;
  if (byte == ' ' || byte == '\t' || byte == '\n')
    return token_space;
  return token_symbol;
}

void tokenize(const char *text, size_t length, struct counts *counts) {
  size_t i = 0;
  while (i < length) {
    enum token_kind kind = classify((unsigned char)text[i]);
    size_t start = i;
    switch (kind) {
    case token_word:
      while (i < length && classify((unsigned char)text[i]) <= token_number)
        ++i;
      break;
    case token_number:
      while (i < length && text[i] >= '0' && text[i] <= '9') {
        counts->digits += (unsigned long)(text[i] - '0');
        ++i;
      }
      break;
    case token_space:
      while (i < length && classify((unsigned char)text[i]) == token_space)
        ++i;
      break;
    case token_symbol:
      ++i;
      break;
    }
    counts->kinds[kind] += 1;
    counts->hash = counts->hash * 31u + (unsigned)(i - start) + kind;
  }
}

size_t count_matches(const char *text, size_t length, const char *word) {
  size_t word_length = strlen(word);
  size_t found = 0;
  for (size_t i = 0; i + word_length <= length; ++i)
    if (memcmp(text + i, word, word_length) == 0)
      ++found;
  return found;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  static const char *const words[] = {"int", "while", "x", "count", "42",
                                      "+",   "(",     ")", "{",     "}",
                                      ";",   "1024",  "=", "value", "_tmp"};
  enum { length = 1 << 16 };
  static char text[length];
  unsigned state = 3;
  size_t used = 0;
  while (used + 8 < length) {
    state = state * 1103515245u + 12345u;
    const char *word = words[(state >> 16) % 15];
    size_t word_length = strlen(word);
    memcpy(text + used, word, word_length);
    used += word_length;
    text[used++] = (state & 3) == 0 ? '\n' : ' ';
  }
  struct counts counts = {{0, 0, 0, 0}, 0, 0};
  size_t matches = 0;
  for (int round = 0; round < 120; ++round) {
    tokenize(text, used, &counts);
    if ((round & 7) == 0)
      matches += count_matches(text, used, "value");
  }
  printf("%u %u %u %u %lu %u %zu\n", counts.kinds[0], counts.kinds[1],
         counts.kinds[2], counts.kinds[3], counts.digits, counts.hash, matches);
  return 0;
}
#endif
