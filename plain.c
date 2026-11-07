// Лабораторная работа №7. Вариант 1: без ручной векторизации.
// Обращение матрицы: A^-1 = (I + R + R^2 + ... + R^(M-1)) * B,
//                    B = A^T / (||A||_1 * ||A||_inf),  R = I - B*A.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BS 256   // размер блока при умножении (чтобы блок помещался в кэш)

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
        A[i] = (float)rand() / RAND_MAX * 2 - 1;  //случайное число от [-1:1]
    for (int i = 0; i < n; i++)
        A[i * n + i] += n / 2.0f;
}


void identity(float *X, int n)  //создание единичной матрицы
{
    memset(X, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        X[i * n + i] = 1;
}

// S = S + P
void add(float *S, const float *P, int n)
{
    for (int i = 0; i < n * n; i++)
        S[i] += P[i];
}

// C = A * B.
void mul(const float *restrict A, const float *restrict B, float *restrict C, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int jj = 0; jj < n; jj += BS)
        for (int kk = 0; kk < n; kk += BS) {
            int j_end = (jj + BS < n) ? jj + BS : n;
            int k_end = (kk + BS < n) ? kk + BS : n;
            for (int i = 0; i < n; i++)
                for (int k = kk; k < k_end; k++) {
                    float a = A[i * n + k];
                    for (int j = jj; j < j_end; j++)
                        C[i * n + j] += a * B[k * n + j];
                }
        }
}

void invert(const float *A, float *X, int n, int m)  // X = A^-1
{
    float *B = new_matrix(n), *R = new_matrix(n), *P = new_matrix(n);
    float *S = new_matrix(n), *T = new_matrix(n);

    // ||A||_1 - максимальная сумма модулей по столбцам, ||A||_inf - по строкам
    float norm1 = 0, norm_inf = 0;
    float *col = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        float row = 0;
        for (int j = 0; j < n; j++) {
            float v = fabsf(A[i * n + j]);
            row += v;
            col[j] += v;
        }
        if (row > norm_inf)
            norm_inf = row;
    }
    for (int j = 0; j < n; j++)
        if (col[j] > norm1)
            norm1 = col[j];
    free(col);

    // B = A^T / (||A||_1 * ||A||_inf)
    float scale = 1 / (norm1 * norm_inf);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i * n + j] = A[j * n + i] * scale;

    // R = I - B*A
    mul(B, A, T, n);
    identity(R, n);
    for (int i = 0; i < n * n; i++)
        R[i] -= T[i];

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

    free(A);
    free(X);
    return 0;
}