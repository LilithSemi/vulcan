unsigned dot_product(const unsigned *left, const unsigned *right,
                     unsigned count) {
  unsigned sum = 0;
  for (unsigned i = 0; i < count; ++i)
    sum += left[i] * right[i];
  return sum;
}

unsigned xor_reduce(const unsigned *values, unsigned count) {
  unsigned checksum = 0;
  for (unsigned i = 0; i < count; ++i)
    checksum ^= values[i] + i * 17u;
  return checksum;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 4096 };
  static unsigned left[count];
  static unsigned right[count];
  unsigned checksum = 0;
  for (unsigned i = 0; i < count; ++i) {
    left[i] = i * 2654435761u + 1013904223u;
    right[i] = (i ^ (i >> 3)) * 2246822519u + 3266489917u;
  }
  for (unsigned iteration = 0; iteration < 4000; ++iteration) {
    checksum += dot_product(left, right, count);
    checksum ^= xor_reduce(right, count);
    left[iteration & (count - 1)] += checksum | 1u;
  }
  printf("%u\n", checksum);
  return 0;
}
#endif
