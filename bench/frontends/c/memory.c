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

void copy_scaled(const int *source, int *destination, unsigned count, int scale) {
    for (unsigned i = 0; i < count; ++i) destination[i] = source[i] * scale + (int)i;
}
