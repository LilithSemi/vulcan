unsigned mix_words(unsigned a, unsigned b, unsigned c, unsigned d) {
  unsigned x = a + b * 3u;
  unsigned y = c ^ (d + 0x9e3779b9u);
  for (unsigned i = 0; i < 64; ++i) {
    x = (x ^ (x << 7)) + (y & 0x55555555u);
    y = (y + (y >> 3)) ^ (x | 0x01010101u);
    x += (a * (i + 1u)) + (b * (64u - i));
  }
  return x ^ y;
}

unsigned polynomial(unsigned x) {
  unsigned x2 = x * x;
  unsigned x4 = x2 * x2;
  return 17u + 31u * x + 7u * x2 + 11u * x2 * x + 3u * x4 + x4 * x;
}

unsigned reductions(const unsigned *values, unsigned count) {
  unsigned sum = 0;
  unsigned bits = 0;
  for (unsigned i = 0; i < count; ++i) {
    unsigned value = values[i];
    sum += value * (i + 3u);
    bits ^= (value & 0x00ff00ffu) | (value >> 16);
  }
  return sum + bits;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  unsigned state = 0x12345678u;
  for (unsigned i = 0; i < 1000000; ++i) {
    state = mix_words(state, i + 1u, state ^ i, 0xabcdef01u - i);
    state ^= polynomial(state + i);
  }
  printf("%u\n", state);
  return 0;
}
#endif
