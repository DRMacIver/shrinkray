/* A C program of the sort fuzzers produce: lots of small helper
 * functions, typedef chains, and one interesting call buried in the
 * middle (the marker here is "overflow_site"). */
#include <stdint.h>
#include <stdlib.h>

typedef unsigned long word_t;
typedef word_t reg_t;

typedef struct VM {
    reg_t regs[16];
    reg_t pc;
    int halted;
} VM;

static reg_t reg_read(VM *vm, int index) { return vm->regs[index & 15]; }

static void reg_write(VM *vm, int index, reg_t value) {
    vm->regs[index & 15] = value;
}

static reg_t add_wrapping(reg_t a, reg_t b) { return a + b; }

static reg_t overflow_site(reg_t a, reg_t b) { return a * b + a; }

static void step(VM *vm) {
    reg_t a = reg_read(vm, 0);
    reg_t b = reg_read(vm, 1);
    reg_write(vm, 2, add_wrapping(a, b));
    reg_write(vm, 3, overflow_site(a, b));
    vm->pc = add_wrapping(vm->pc, 1);
}

int main(void) {
    VM *vm = calloc(1, sizeof(VM));
    if (vm == NULL) {
        return 1;
    }
    reg_write(vm, 0, 0x7fffffffUL);
    reg_write(vm, 1, 3);
    while (!vm->halted && vm->pc < 4) {
        step(vm);
    }
    free(vm);
    return 0;
}
