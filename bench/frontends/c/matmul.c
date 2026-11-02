enum { size = 128 };

void matmul_int(int out[size][size], const int left[size][size],
                const int right[size][size]) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j) {
      int sum = 0;
      for (int k = 0; k < size; ++k)
        sum += left[i][k] * right[k][j];
      out[i][j] = sum;
    }
}

void matmul_float(float out[size][size], const float left[size][size],
                  const float right[size][size]) {
  for (int i = 0; i < size; ++i)
    for (int k = 0; k < size; ++k) {
      float scale = left[i][k];
      for (int j = 0; j < size; ++j)
        out[i][j] += scale * right[k][j];
    }
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

static int int_left[size][size], int_right[size][size], int_out[size][size];
static float float_left[size][size], float_right[size][size],
    float_out[size][size];

int main(void) {
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j) {
      int_left[i][j] = (i * 3 + j) & 15;
      int_right[i][j] = (i ^ j) & 7;
      float_left[i][j] = (float)((i + j) & 7) * 0.5f;
      float_right[i][j] = (float)((i * j) & 3);
    }
  long total = 0;
  double float_total = 0;
  for (int round = 0; round < 12; ++round) {
    matmul_int(int_out, int_left, int_right);
    matmul_float(float_out, float_left, float_right);
    total += int_out[round][round + 1];
    float_total += float_out[round][round];
    int_left[round][round] += 1;
  }
  printf("%ld %.1f\n", total, float_total);
  return 0;
}
#endif
