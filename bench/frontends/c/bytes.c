#include <stdint.h>

uint32_t fnv1a(const uint8_t *bytes, uint32_t count) {
  uint32_t hash = 2166136261u;
  for (uint32_t i = 0; i < count; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

uint32_t count_byte(const uint8_t *bytes, uint32_t count, uint8_t needle) {
  uint32_t found = 0;
  for (uint32_t i = 0; i < count; ++i)
    found += bytes[i] == needle;
  return found;
}

uint32_t crc32(const uint8_t *bytes, uint32_t count) {
  uint32_t crc = 0xffffffffu;
  for (uint32_t i = 0; i < count; ++i) {
    crc ^= bytes[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

void to_upper(uint8_t *bytes, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    if (bytes[i] >= 'a' && bytes[i] <= 'z')
      bytes[i] -= 32;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 16384 };
  static uint8_t buffer[count];
  uint32_t state = 7, checksum = 0;
  for (uint32_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    buffer[i] = (uint8_t)(32 + (state >> 24) % 95);
  }
  for (uint32_t iteration = 0; iteration < 2000; ++iteration) {
    checksum += fnv1a(buffer, count);
    checksum ^= count_byte(buffer, count, ' ');
    if ((iteration & 63) == 0)
      checksum += crc32(buffer, count);
    buffer[iteration & (count - 1)] = (uint8_t)('a' + iteration % 26);
    if ((iteration & 255) == 0)
      to_upper(buffer, count);
  }
  printf("%u\n", checksum);
  return 0;
}
#endif
