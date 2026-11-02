float dot_float(const float *left, const float *right, unsigned count) {
  float sum = 0;
  for (unsigned i = 0; i < count; ++i)
    sum += left[i] * right[i];
  return sum;
}

void saxpy(float *y, const float *x, float scale, unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    y[i] = scale * x[i] + y[i];
}

double horner(double x) {
  return (((3.0 * x + 2.0) * x - 5.0) * x + 1.5) * x - 0.25;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 4096 };
  static float left[count];
  static float right[count];
  for (unsigned i = 0; i < count; ++i) {
    left[i] = (float)(i % 97) * 0.01f;
    right[i] = (float)(i % 89) * 0.02f;
  }
  double total = 0;
  for (unsigned iteration = 0; iteration < 6000; ++iteration) {
    total += dot_float(left, right, count);
    saxpy(right, left, 0.0001f, count);
    total += horner((double)(iteration & 255) * 0.001);
  }
  printf("%.3f\n", total);
  return 0;
}
#endif
