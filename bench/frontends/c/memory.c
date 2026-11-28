struct particle {
  int x;
  int y;
  int z;
  int velocity;
};

void update_particles(struct particle *particles, unsigned count, int dt) {
  for (unsigned i = 0; i < count; ++i) {
    struct particle *particle = &particles[i];
    particle->x += particle->velocity * dt;
    particle->y += (particle->velocity + (int)i) * dt;
    particle->z += (particle->velocity - (int)i) * dt;
    particle->velocity += (particle->x ^ particle->y) & 7;
  }
}

unsigned stencil(const unsigned *input, unsigned *output, unsigned count) {
  unsigned checksum = 0;
  for (unsigned i = 1; i + 1 < count; ++i) {
    unsigned value = input[i - 1] + input[i] * 2u + input[i + 1];
    output[i] = value;
    checksum ^= value;
  }
  return checksum;
}

void copy_scaled(const unsigned *source, unsigned *destination, unsigned count,
                 unsigned scale) {
  for (unsigned i = 0; i < count; ++i)
    destination[i] = source[i] * scale + i;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 4096 };
  static struct particle particles[count];
  static unsigned source[count];
  static unsigned scratch[count];
  unsigned checksum = 0;
  for (unsigned i = 0; i < count; ++i) {
    particles[i].x = (int)i;
    particles[i].y = (int)(i * 3u);
    particles[i].z = (int)(i * 7u);
    particles[i].velocity = (int)(i & 31u) - 16;
    source[i] = (i & 255u) - 128u;
  }
  for (unsigned iteration = 0; iteration < 4000; ++iteration) {
    update_particles(particles, count, (int)(iteration & 3u) + 1);
    copy_scaled(source, scratch, count, (iteration & 7u) + 1u);
    checksum ^= stencil(scratch, source, count);
  }
  checksum ^= particles[17].x ^ particles[2049].velocity ^ source[1024];
  printf("%u\n", checksum);
  return 0;
}
#endif
