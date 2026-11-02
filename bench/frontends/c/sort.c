void insertion_sort(int *values, int count) {
  for (int i = 1; i < count; ++i) {
    int value = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > value) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = value;
  }
}

void quick_sort(int *values, int low, int high) {
  while (low < high) {
    int pivot = values[(low + high) >> 1];
    int i = low, j = high;
    while (i <= j) {
      while (values[i] < pivot)
        ++i;
      while (values[j] > pivot)
        --j;
      if (i <= j) {
        int swap = values[i];
        values[i] = values[j];
        values[j] = swap;
        ++i;
        --j;
      }
    }
    if (j - low < high - i) {
      quick_sort(values, low, j);
      low = i;
    } else {
      quick_sort(values, i, high);
      high = j;
    }
  }
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 20000 };
  static int large[count];
  static int small[256];
  unsigned state = 12345, checksum = 0;
  for (int round = 0; round < 60; ++round) {
    for (int i = 0; i < count; ++i) {
      state = state * 1103515245u + 12345u;
      large[i] = (int)(state >> 8);
    }
    quick_sort(large, 0, count - 1);
    checksum += (unsigned)large[round * 7] ^ (unsigned)large[count - 1 - round];
  }
  for (int round = 0; round < 4000; ++round) {
    for (int i = 0; i < 256; ++i) {
      state = state * 1103515245u + 12345u;
      small[i] = (int)(state >> 20);
    }
    insertion_sort(small, 256);
    checksum += (unsigned)small[round & 255];
  }
  printf("%u\n", checksum);
  return 0;
}
#endif
