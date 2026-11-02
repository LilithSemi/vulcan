#include <stdint.h>
#include <string.h>

enum { slot_count = 1 << 14 };

struct table {
  uint32_t keys[slot_count];
  uint32_t values[slot_count];
};

static uint32_t hash_key(uint32_t key) {
  key ^= key >> 16;
  key *= 0x7feb352du;
  key ^= key >> 15;
  return key;
}

void table_clear(struct table *table) {
  memset(table->keys, 0, sizeof table->keys);
}

void table_add(struct table *table, uint32_t key, uint32_t value) {
  uint32_t slot = hash_key(key) & (slot_count - 1);
  while (table->keys[slot] != 0 && table->keys[slot] != key)
    slot = (slot + 1) & (slot_count - 1);
  if (table->keys[slot] == key) {
    table->values[slot] += value;
  } else {
    table->keys[slot] = key;
    table->values[slot] = value;
  }
}

uint32_t table_get(const struct table *table, uint32_t key) {
  uint32_t slot = hash_key(key) & (slot_count - 1);
  while (table->keys[slot] != 0) {
    if (table->keys[slot] == key)
      return table->values[slot];
    slot = (slot + 1) & (slot_count - 1);
  }
  return 0;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  static struct table table;
  uint32_t state = 1, checksum = 0;
  for (int round = 0; round < 300; ++round) {
    table_clear(&table);
    for (int i = 0; i < 10000; ++i) {
      state = state * 1664525u + 1013904223u;
      table_add(&table, (state >> 18) + 1, state & 255);
    }
    for (uint32_t key = 1; key < 12000; key += 3)
      checksum += table_get(&table, key) * key;
  }
  printf("%u\n", checksum);
  return 0;
}
#endif
