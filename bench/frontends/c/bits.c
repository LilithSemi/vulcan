#include <stdint.h>

uint64_t splitmix(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

uint64_t popcount(uint64_t value) {
  uint64_t count = 0;
  while (value) {
    value &= value - 1;
    ++count;
  }
  return count;
}

uint64_t mix_step(uint64_t *state, uint64_t total) {
  uint64_t value = splitmix(state);
  total += value % 1000003u;
  return total ^ popcount(value & 0xffff);
}

#ifdef BENCH_EXECUTE
#include <inttypes.h>
#include <stdio.h>

int main(void) {
  uint64_t state = 1, total = 0;
  for (uint32_t i = 0; i < 10000000; ++i)
    total = mix_step(&state, total);
  printf("%" PRIu64 "\n", total);
  return 0;
}
#endif
