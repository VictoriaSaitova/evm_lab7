// Лабораторная работа №7. Вариант 3: матричные операции из библиотеки BLAS.
// Обращение матрицы: A^-1 = (I + R + R^2 + ... + R^(M-1)) * B,
//                    B = A^T / (||A||_1 * ||A||_inf),  R = I - B*A.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

float *new_matrix(int n)
{
    float *p = calloc((size_t)n * n, sizeof(float));
    if (p == NULL) {
        printf("Not enough memory\n");
        exit(1);
    }
    return p;
}

// Случайные числа из [-1, 1] + N/2 на диагонали (диагональное преобладание)
void fill_matrix(float *A, int n)
{
    srand(12345);
    for (int i = 0; i < n * n; i++)
        A[i] = (float)rand() / RAND_MAX * 2 - 1;
    for (int i = 0; i < n; i++)
        A[i * n + i] += n / 2.0f;
}

// X = I
void identity(float *X, int n)
{
    memset(X, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        X[i * n + i] = 1;
}

// S = S + P
void add(float *S, const float *P, int n)
{
    cblas_saxpy(n * n, 1.0f, P, 1, S, 1);
}

// C = A * B
void mul(const float *A, const float *B, float *C, int n)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                1.0f, A, n, B, n, 0.0f, C, n);
}

// X = A^-1 (приближённо, M членов ряда)
void invert(const float *A, float *X, int n, int m)
{
    float *B = new_matrix(n), *R = new_matrix(n), *P = new_matrix(n);
    float *S = new_matrix(n), *T = new_matrix(n);

    // ||A||_1 - максимальная сумма модулей по столбцам, ||A||_inf - по строкам
    float norm1 = 0, norm_inf = 0;
    for (int i = 0; i < n; i++) {
        float row = cblas_sasum(n, &A[i * n], 1);   // сумма |x| по строке i (шаг 1)
        float col = cblas_sasum(n, &A[i], n);       // сумма |x| по столбцу i (шаг n)
        if (row > norm_inf) norm_inf = row;
        if (col > norm1) norm1 = col;
    }

    // B = A^T / (||A||_1 * ||A||_inf)
    float scale = 1 / (norm1 * norm_inf);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i * n + j] = A[j * n + i] * scale;

    // R = I - B*A
    mul(B, A, T, n);
    identity(R, n);
    cblas_saxpy(n * n, -1.0f, T, 1, R, 1);   // R = R - T

    // S = I + R + R^2 + ... + R^(m-1)
    identity(S, n);
    memcpy(P, R, (size_t)n * n * sizeof(float));   // P = R
    if (m >= 2)
        add(S, P, n);
    for (int k = 2; k < m; k++) {
        mul(P, R, T, n);                  // T = P * R
        float *tmp = P; P = T; T = tmp;   // P = T (меняем указатели, без копирования)
        add(S, P, n);
    }

    // A^-1 = S * B
    mul(S, B, X, n);

    free(B); free(R); free(P); free(S); free(T);
}

// max |A*X - I| (проверка: если X = A^-1, то A*X = I)
float check(const float *A, const float *X, int n)
{
    float *T = new_matrix(n);
    mul(A, X, T, n);
    float err = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            err = fmaxf(err, fabsf(T[i * n + j] - (i == j)));
    free(T);
    return err;
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 2048;   // N - размер матрицы
    int m = (argc > 2) ? atoi(argv[2]) : 10;     // M - число членов ряда

    float *A = new_matrix(n), *X = new_matrix(n);
    fill_matrix(A, n);

    clock_t start = clock();
    invert(A, X, n, m);
    double time = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("N = %d, M = %d, time = %.3f s\n", n, m, time);
    printf("max |A*X - I| = %e\n", check(A, X, n));

    free(A); free(X);
    return 0;
}