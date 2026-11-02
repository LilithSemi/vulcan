struct node {
  struct node *next;
  int value;
  int weight;
};

long walk_list(const struct node *node) {
  long sum = 0;
  while (node) {
    sum += (long)node->value * node->weight;
    node = node->next;
  }
  return sum;
}

unsigned fib(unsigned n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  enum { count = 50000 };
  static struct node nodes[count];
  static int order[count];
  unsigned state = 99;
  for (int i = 0; i < count; ++i)
    order[i] = i;
  for (int i = count - 1; i > 0; --i) {
    state = state * 1103515245u + 12345u;
    int j = (int)((state >> 8) % (unsigned)(i + 1));
    int swap = order[i];
    order[i] = order[j];
    order[j] = swap;
  }
  for (int i = 0; i < count; ++i) {
    nodes[order[i]].next = i + 1 < count ? &nodes[order[i + 1]] : 0;
    nodes[i].value = i & 1023;
    nodes[i].weight = (i * 7) & 63;
  }
  long total = 0;
  for (int round = 0; round < 400; ++round)
    total += walk_list(&nodes[order[0]]);
  total += fib(30);
  printf("%ld\n", total);
  return 0;
}
#endif
