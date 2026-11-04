#include <stddef.h>
#include <stdint.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

enum { max_files = 64, buffer_size = 256, log_size = 1024 };
enum { op_read, op_write, op_seek, op_stat, op_count };
enum { err_badf = 9, err_acces = 13, err_fault = 14, err_inval = 22, err_nospc = 28 };

struct file {
  uint32_t mode;
  uint32_t owner;
  uint32_t pos;
  uint32_t size;
  uint8_t data[buffer_size];
};

struct task {
  uint32_t uid;
  uint32_t calls;
  uint32_t errors;
  struct file *files[max_files];
};

struct log {
  char text[log_size];
  uint32_t head;
  uint32_t dropped;
};

static struct log klog;

static void log_put(char c) {
  if (unlikely(klog.head >= log_size)) {
    klog.dropped++;
    klog.head = 0;
  }
  klog.text[klog.head++] = c;
}

static void log_number(uint32_t value) {
  char digits[10];
  int count = 0;
  do {
    digits[count++] = (char)('0' + value % 10);
    value /= 10;
  } while (value != 0);
  while (count > 0)
    log_put(digits[--count]);
}

/* Error reports run rarely but are large, like the kernel's pr_err and WARN_ON paths. */
static inline long report(struct task *task, const char *what, uint32_t a, uint32_t b, int err) {
  task->errors++;
  for (const char *p = what; *p; ++p)
    log_put(*p);
  log_put(' ');
  log_number(a);
  log_put(',');
  log_number(b);
  log_put(' ');
  log_number((uint32_t)err);
  log_put('\n');
  return -err;
}

static long do_read(struct task *task, struct file *file, uint8_t *out, uint32_t len) {
  if (unlikely(!(file->mode & 4)))
    return report(task, "read: not readable", file->owner, file->mode, err_acces);
  if (unlikely(out == NULL))
    return report(task, "read: bad buffer", len, file->pos, err_fault);
  uint32_t avail = file->size - file->pos;
  if (len > avail)
    len = avail;
  uint32_t sum = 0;
  for (uint32_t i = 0; i < len; ++i) {
    uint8_t byte = file->data[file->pos + i];
    out[i] = byte;
    sum += byte;
  }
  file->pos += len;
  return (long)(sum + len);
}

static long do_write(struct task *task, struct file *file, const uint8_t *in, uint32_t len) {
  if (unlikely(!(file->mode & 2)))
    return report(task, "write: not writable", file->owner, file->mode, err_acces);
  if (unlikely(in == NULL))
    return report(task, "write: bad buffer", len, file->pos, err_fault);
  if (unlikely(file->pos + len > buffer_size))
    return report(task, "write: no space", file->pos, len, err_nospc);
  for (uint32_t i = 0; i < len; ++i)
    file->data[file->pos + i] = (uint8_t)(in[i] ^ 0x5a);
  file->pos += len;
  if (file->pos > file->size)
    file->size = file->pos;
  return (long)len;
}

static long do_seek(struct task *task, struct file *file, uint32_t offset) {
  if (unlikely(offset > file->size))
    return report(task, "seek: past end", offset, file->size, err_inval);
  file->pos = offset;
  return (long)offset;
}

static long do_stat(struct file *file) {
  return (long)(file->size * 3u + file->mode + file->owner);
}

long dispatch(struct task *task, uint32_t op, uint32_t fd, uint32_t arg, uint8_t *buffer) {
  task->calls++;
  if (unlikely(op >= op_count))
    return report(task, "dispatch: bad op", op, fd, err_inval);
  if (unlikely(fd >= max_files))
    return report(task, "dispatch: bad fd", fd, op, err_badf);
  struct file *file = task->files[fd];
  if (unlikely(file == NULL))
    return report(task, "dispatch: closed fd", fd, op, err_badf);
  if (unlikely(file->owner != task->uid && !(file->mode & 1)))
    return report(task, "dispatch: denied", task->uid, file->owner, err_acces);
  switch (op) {
  case op_read:
    return do_read(task, file, buffer, arg & 31);
  case op_write:
    return do_write(task, file, buffer, arg & 15);
  case op_seek:
    return do_seek(task, file, arg & 127);
  default:
    return do_stat(file);
  }
}

#ifdef BENCH_EXECUTE
#include <stdio.h>

int main(void) {
  static struct file files[max_files];
  static struct task task;
  static uint8_t buffer[64];
  task.uid = 1000;
  for (int i = 0; i < max_files; ++i) {
    files[i].mode = (i % 7 == 0) ? 5u : 7u;
    files[i].owner = (i % 11 == 0) ? 0u : 1000u;
    files[i].size = 128;
    for (int k = 0; k < buffer_size; ++k)
      files[i].data[k] = (uint8_t)(k * 13 + i);
    task.files[i] = (i % 29 == 28) ? NULL : &files[i];
  }
  for (int k = 0; k < 64; ++k)
    buffer[k] = (uint8_t)(k * 7);

  uint32_t state = 12345;
  long total = 0;
  for (int round = 0; round < 6000000; ++round) {
    state = state * 1664525u + 1013904223u;
    uint32_t op = (state >> 8) & 3;
    uint32_t fd = (state >> 12) & 63;
    uint32_t arg = state >> 20;
    /* About one call in 250 asks for something invalid. */
    if ((state & 0xff) == 0)
      op = 7;
    total += dispatch(&task, op, fd, arg, buffer);
  }
  printf("%ld %u %u %u\n", total, task.calls, task.errors, klog.dropped);
  return 0;
}
#endif
