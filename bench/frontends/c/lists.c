#include <stddef.h>
#include <stdint.h>

struct link {
  struct link *next;
  struct link *prev;
};

struct entry {
  struct link lru;
  struct entry *chain;
  uint32_t key;
  uint32_t value;
};

struct item {
  struct item *next;
  uint32_t key;
};

enum { capacity = 4096, bucket_count = 1024 };

struct cache {
  struct link lru;
  struct entry *buckets[bucket_count];
  struct entry *free;
  struct entry pool[capacity];
  uint32_t hits;
};

#define entry_of(l)                                                            \
  ((struct entry *)((char *)(l) - offsetof(struct entry, lru)))

static void link_remove(struct link *l) {
  l->prev->next = l->next;
  l->next->prev = l->prev;
}

static void link_push(struct link *head, struct link *l) {
  l->next = head->next;
  l->prev = head;
  head->next->prev = l;
  head->next = l;
}

void cache_init(struct cache *c) {
  c->lru.next = c->lru.prev = &c->lru;
  for (int i = 0; i < bucket_count; ++i)
    c->buckets[i] = 0;
  c->free = 0;
  for (int i = capacity - 1; i >= 0; --i) {
    c->pool[i].chain = c->free;
    c->free = &c->pool[i];
  }
  c->hits = 0;
}

static void chain_unlink(struct cache *c, struct entry *e) {
  struct entry **at = &c->buckets[e->key & (bucket_count - 1)];
  while (*at != e)
    at = &(*at)->chain;
  *at = e->chain;
}

uint32_t cache_get(struct cache *c, uint32_t key) {
  struct entry **bucket = &c->buckets[key & (bucket_count - 1)];
  for (struct entry *e = *bucket; e; e = e->chain) {
    if (e->key == key) {
      link_remove(&e->lru);
      link_push(&c->lru, &e->lru);
      c->hits++;
      return e->value;
    }
  }
  struct entry *e = c->free;
  if (e) {
    c->free = e->chain;
  } else {
    e = entry_of(c->lru.prev);
    link_remove(&e->lru);
    chain_unlink(c, e);
  }
  e->key = key;
  e->value = key * 2654435761u;
  e->chain = *bucket;
  *bucket = e;
  link_push(&c->lru, &e->lru);
  return e->value;
}

static struct item *merge(struct item *a, struct item *b) {
  struct item head;
  struct item *tail = &head;
  while (a && b) {
    if (a->key <= b->key) {
      tail->next = a;
      a = a->next;
    } else {
      tail->next = b;
      b = b->next;
    }
    tail = tail->next;
  }
  tail->next = a ? a : b;
  return head.next;
}

struct item *list_sort(struct item *list) {
  struct item *parts[32] = {0};
  while (list) {
    struct item *one = list;
    list = list->next;
    one->next = 0;
    int i = 0;
    while (parts[i]) {
      one = merge(parts[i], one);
      parts[i] = 0;
      ++i;
    }
    parts[i] = one;
  }
  struct item *out = 0;
  for (int i = 0; i < 32; ++i)
    if (parts[i])
      out = merge(parts[i], out);
  return out;
}

struct item *list_reverse(struct item *list) {
  struct item *out = 0;
  while (list) {
    struct item *next = list->next;
    list->next = out;
    out = list;
    list = next;
  }
  return out;
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  static struct cache cache;
  static struct item items[20000];
  uint32_t state = 7, checksum = 0;
  cache_init(&cache);
  for (int i = 0; i < 2000000; ++i) {
    state = state * 1664525u + 1013904223u;
    uint32_t key = (state >> 20) & 8191;
    if (state & 0x80000)
      key &= 2047;
    checksum += cache_get(&cache, key);
  }
  checksum += cache.hits;
  for (int round = 0; round < 40; ++round) {
    struct item *list = 0;
    for (int i = 0; i < 20000; ++i) {
      state = state * 1664525u + 1013904223u;
      items[i].key = state >> 8;
      items[i].next = list;
      list = &items[i];
    }
    list = list_reverse(list_sort(list));
    uint32_t prev = 0xffffffffu;
    for (struct item *it = list; it; it = it->next) {
      if (it->key > prev)
        checksum += 1;
      checksum = checksum * 31 + it->key;
      prev = it->key;
    }
  }
  printf("%u\n", checksum);
  return 0;
}
#endif
