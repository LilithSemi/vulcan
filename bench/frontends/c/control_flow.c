int classify(int value) {
  switch (value & 15) {
  case 0:
    return value + 17;
  case 1:
    return value * 3;
  case 2:
    return value - 29;
  case 3:
    return value ^ 0x55aa;
  case 4:
    return value > 100 ? value / 5 : value * 5;
  case 5:
    return value < 0 ? -value : value;
  case 6:
    return (value & 1) ? value + 1 : value - 1;
  case 7:
    return value | 0x1010;
  case 8:
    return value & 0x7fff;
  case 9:
    return value + (value >> 2);
  case 10:
    return value - (value >> 3);
  case 11:
    return value == 11 ? 42 : 13;
  case 12:
    return value != 12 ? 19 : 23;
  case 13:
    return value >= 13 ? 31 : 37;
  case 14:
    return value <= 14 ? 41 : 43;
  default:
    return value ^ (value << 5);
  }
}

int decision_tree(int a, int b, int c) {
  int result = 0;
  if (a > b) {
    result = c > a ? c - b : a - b;
    if ((result & 3) == 0)
      result += classify(result);
  } else {
    result = c < b ? b - c : c - a;
    if ((result & 7) == 3)
      result -= classify(result);
  }
  return result > 1000 ? 1000 : (result < -1000 ? -1000 : result);
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  unsigned state = 0x31415926u;
  unsigned checksum = 0;
  for (unsigned i = 0; i < 10000000; ++i) {
    state = state * 1664525u + 1013904223u;
    int a = (int)(state & 2047u) - 1024;
    int b = (int)((state >> 11) & 2047u) - 1024;
    int c = (int)((state >> 21) & 1023u) - 512;
    checksum ^= (unsigned)classify(a + (int)(i & 15u));
    checksum += (unsigned)(b + c);
  }
  printf("%u\n", checksum);
  return 0;
}
#endif
