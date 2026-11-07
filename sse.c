#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <xmmintrin.h>   // SSE: тип __m128 (4 числа float) и функции _mm_...

#define BS 256   // размер блока при умножении

float *new_matrix(int n)
{
    float *p = calloc((size_t)n * n, sizeof(float));
    if (p == NULL) {
        printf("Not enough memory\n");
        exit(1);
    }
    return p;
}


void fill_matrix(float *A, int n)
{
    srand(12345);
    for (int i = 0; i < n * n; i++)
        A[i] = (float)rand() / RAND_MAX * 2 - 1;
    for (int i = 0; i < n; i++)
        A[i * n + i] += n / 2.0f;
}


void identity(float *X, int n)
{
    memset(X, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        X[i * n + i] = 1;
}

// S = S + P, по 4 элемента за команду
void add(float *S, const float *P, int n)
{
    int i = 0;
    for (; i + 4 <= n * n; i += 4)
        _mm_storeu_ps(&S[i], _mm_add_ps(_mm_loadu_ps(&S[i]), _mm_loadu_ps(&P[i])));
    for (; i < n * n; i++)       // остаток, если n*n не делится на 4
        S[i] += P[i];
}

/*C = A * B. Как в варианте 1 (i-k-j + блоки), но внутренний цикл обрабатывает по 4 элемента строки за одну SSE-команду
за один проход к строке C прибавляются сразу 4 строки B (k, k+1, k+2, k+3),
поэтому C загружается и записывается в 4 раза реже*/
void mul(const float *A, const float *B, float *C, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int jj = 0; jj < n; jj += BS)
        for (int kk = 0; kk < n; kk += BS) {
            int j_end = (jj + BS < n) ? jj + BS : n;
            int k_end = (kk + BS < n) ? kk + BS : n;
            for (int i = 0; i < n; i++) {
                float *c = &C[i * n];               // строка i матрицы C
                const float *a = &A[i * n];         // строка i матрицы A
                int k = kk;
                for (; k + 4 <= k_end; k += 4) {
                    __m128 a0 = _mm_set1_ps(a[k]);      // (A[i][k],   x4)
                    __m128 a1 = _mm_set1_ps(a[k + 1]);  // (A[i][k+1], x4)
                    __m128 a2 = _mm_set1_ps(a[k + 2]);
                    __m128 a3 = _mm_set1_ps(a[k + 3]);
                    const float *b0 = &B[k * n], *b1 = b0 + n, *b2 = b1 + n, *b3 = b2 + n;
                    int j = jj;
                    for (; j + 4 <= j_end; j += 4) {
                        __m128 cv = _mm_loadu_ps(&c[j]);
                        cv = _mm_add_ps(cv, _mm_mul_ps(a0, _mm_loadu_ps(&b0[j])));
                        cv = _mm_add_ps(cv, _mm_mul_ps(a1, _mm_loadu_ps(&b1[j])));
                        cv = _mm_add_ps(cv, _mm_mul_ps(a2, _mm_loadu_ps(&b2[j])));
                        cv = _mm_add_ps(cv, _mm_mul_ps(a3, _mm_loadu_ps(&b3[j])));
                        _mm_storeu_ps(&c[j], cv);
                    }
                    for (; j < j_end; j++)              // остаток (< 4 столбцов)
                        c[j] += a[k] * b0[j] + a[k + 1] * b1[j] + a[k + 2] * b2[j] + a[k + 3] * b3[j];
                }
                for (; k < k_end; k++)                  // остаток (< 4 строк B)
                    for (int j = jj; j < j_end; j++)
                        c[j] += a[k] * B[k * n + j];
            }
        }
}

// X = A^-1 (приближённо, M членов ряда)
void invert(const float *A, float *X, int n, int m)
{
    float *B = new_matrix(n), *R = new_matrix(n), *P = new_matrix(n);
    float *S = new_matrix(n), *T = new_matrix(n);

    // ||A||_1 - максимальная сумма модулей по столбцам, ||A||_inf - по строкам
    float norm1 = 0, norm_inf = 0;
    float *col = calloc(n, sizeof(float));
    __m128 sign = _mm_set1_ps(-0.0f);     // маска знакового бита: |x| = x без знака
    for (int i = 0; i < n; i++) {
        __m128 rs = _mm_setzero_ps();     // 4 частичные суммы строки
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            __m128 v = _mm_andnot_ps(sign, _mm_loadu_ps(&A[i * n + j]));   // |A|
            rs = _mm_add_ps(rs, v);
            _mm_storeu_ps(&col[j], _mm_add_ps(_mm_loadu_ps(&col[j]), v));
        }
        float part[4];
        _mm_storeu_ps(part, rs);
        float row = part[0] + part[1] + part[2] + part[3];
        for (; j < n; j++) {              // остаток
            float v = fabsf(A[i * n + j]);
            row += v;
            col[j] += v;
        }
        if (row > norm_inf) norm_inf = row;
    }
    for (int j = 0; j < n; j++)
        if (col[j] > norm1) norm1 = col[j];
    free(col);

    // B = A^T / (||A||_1 * ||A||_inf)
    float scale = 1 / (norm1 * norm_inf);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i * n + j] = A[j * n + i] * scale;

    // R = I - B*A
    mul(B, A, T, n);
    identity(R, n);
    int i = 0;
    for (; i + 4 <= n * n; i += 4)        // R = R - T, по 4 элемента
        _mm_storeu_ps(&R[i], _mm_sub_ps(_mm_loadu_ps(&R[i]), _mm_loadu_ps(&T[i])));
    for (; i < n * n; i++)
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

    free(A); free(X);
    return 0;
}