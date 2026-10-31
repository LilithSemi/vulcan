#include <stdint.h>

extern "C" __global__ void vector_mix(uint32_t *out, const uint32_t *a, const uint32_t *b) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = a[i] * 1664525u + b[i];
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    out[i] = x;
}
