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

int stencil(const int *input, int *output, unsigned count) {
  int checksum = 0;
  for (unsigned i = 1; i + 1 < count; ++i) {
    int value = input[i - 1] + input[i] * 2 + input[i + 1];
    output[i] = value;
    checksum ^= value;
  }
  return checksum;
}

void copy_scaled(const int *source, int *destination, unsigned count,
                 int scale) {
  for (unsigned i = 0; i < count; ++i)
    destination[i] = source[i] * scale + (int)i;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 4096 };
  static struct particle particles[count];
  static int source[count];
  static int scratch[count];
  int checksum = 0;
  for (unsigned i = 0; i < count; ++i) {
    particles[i].x = (int)i;
    particles[i].y = (int)(i * 3u);
    particles[i].z = (int)(i * 7u);
    particles[i].velocity = (int)(i & 31u) - 16;
    source[i] = (int)(i & 255u) - 128;
  }
  for (unsigned iteration = 0; iteration < 4000; ++iteration) {
    update_particles(particles, count, (int)(iteration & 3u) + 1);
    copy_scaled(source, scratch, count, (int)(iteration & 7u) + 1);
    checksum ^= stencil(scratch, source, count);
  }
  checksum ^= particles[17].x ^ particles[2049].velocity ^ source[1024];
  printf("%d\n", checksum);
  return 0;
}
#endif
