#include <stdio.h>
#include <stdlib.h>

#define ROWS 3
#define COLS 3

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

static double get(const Matrix *m, int r, int c) {
    return m->data[r * m->cols + c];
}

static void set(Matrix *m, int r, int c, double value) {
    m->data[r * m->cols + c] = value;
}

static Matrix *matrix_new(int rows, int cols) {
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = calloc((size_t)(rows * cols), sizeof(double));
    return m;
}

static double trace(const Matrix *m) {
    double total = 0.0;
    for (int i = 0; i < m->rows && i < m->cols; i++) {
        total += get(m, i, i);
    }
    return total;
}

int main(void) {
    Matrix *m = matrix_new(ROWS, COLS);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            set(m, i, j, (double)(i * COLS + j));
        }
    }
    printf("trace = %f\n", trace(m));
    free(m->data);
    free(m);
    return 0;
}
