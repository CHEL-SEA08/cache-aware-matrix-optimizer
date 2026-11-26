
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <fstream>
#include <iomanip>

using namespace std;

// 1. Naive Matrix Multiplication
void naive(vector<float>& A, vector<float>& B,
           vector<float>& C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

// 2. Loop Reordered
void loopReordered(vector<float>& A, vector<float>& B,
                   vector<float>& C, int N)
{
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
        {
            float temp = A[i * N + k];

            for (int j = 0; j < N; j++)
                C[i * N + j] += temp * B[k * N + j];
        }
}

// 3. Cache Blocked
void blocked(vector<float>& A, vector<float>& B,
             vector<float>& C, int N, int BS)
{
    for (int ii = 0; ii < N; ii += BS)
        for (int kk = 0; kk < N; kk += BS)
            for (int jj = 0; jj < N; jj += BS)
                for (int i = ii; i < min(ii + BS, N); i++)
                    for (int k = kk; k < min(kk + BS, N); k++)
                    {
                        float temp = A[i * N + k];

                        for (int j = jj; j < min(jj + BS, N); j++)
                            C[i * N + j] += temp * B[k * N + j];
                    }
}

// 4. AVX2 SIMD
void simdAVX2(vector<float>& A, vector<float>& B,
              vector<float>& C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            __m256 a = _mm256_set1_ps(A[i * N + k]);

            int j = 0;

            for (; j + 7 < N; j += 8)
            {
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c = _mm256_loadu_ps(&C[i * N + j]);

                c = _mm256_fmadd_ps(a, b, c);

                _mm256_storeu_ps(&C[i * N + j], c);
            }

            for (; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
        }
    }
}

// 5. OpenMP
void parallelMatrix(vector<float>& A, vector<float>& B,
                    vector<float>& C, int N)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float temp = A[i * N + k];

            for (int j = 0; j < N; j++)
                C[i * N + j] += temp * B[k * N + j];
        }
    }
}

// Measure execution time
template <typename Function>
double measureTime(Function function)
{
    auto start = chrono::high_resolution_clock::now();

    function();

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    return elapsed.count();
}

// Benchmark one matrix size
void benchmark(int N, ofstream& csv)
{
    int BS = 32;

    vector<float> A(N * N, 1.0f);
    vector<float> B(N * N, 2.0f);

    vector<float> C1(N * N, 0.0f);
    vector<float> C2(N * N, 0.0f);
    vector<float> C3(N * N, 0.0f);
    vector<float> C4(N * N, 0.0f);
    vector<float> C5(N * N, 0.0f);

    double t1 = measureTime([&]() {
        naive(A, B, C1, N);
    });

    double t2 = measureTime([&]() {
        loopReordered(A, B, C2, N);
    });

    double t3 = measureTime([&]() {
        blocked(A, B, C3, N, BS);
    });

    double t4 = measureTime([&]() {
        simdAVX2(A, B, C4, N);
    });

    double t5 = measureTime([&]() {
        parallelMatrix(A, B, C5, N);
    });

    double operations = 2.0 * N * N * N;

    double times[] = {t1, t2, t3, t4, t5};

    string names[] = {
        "Naive",
        "Loop Reordered",
        "Cache Blocked",
        "AVX2 SIMD",
        "OpenMP"
    };

    cout << "\n====================================\n";
    cout << "Matrix Size: " << N << " x " << N << endl;
    cout << "====================================\n";

    cout << "Results: "
         << C1[0] << ", "
         << C2[0] << ", "
         << C3[0] << ", "
         << C4[0] << ", "
         << C5[0] << endl;

    cout << "\nAlgorithm\t\tTime(s)\t\tGFLOPS\t\tSpeedup\n";

    for (int i = 0; i < 5; i++)
    {
        double gflops = operations / times[i] / 1e9;
        double speedup = t1 / times[i];

        cout << left << setw(20) << names[i]
             << fixed << setprecision(6)
             << times[i] << "\t"
             << gflops << "\t"
             << speedup << "x\n";

        csv << N << ","
            << names[i] << ","
            << times[i] << ","
            << gflops << ","
            << speedup << "\n";
    }
}

int main()
{
    int sizes[] = {128, 256, 512, 1024};

    ofstream csv("results/benchmark.csv");

    if (!csv.is_open())
    {
        cout << "Error: Could not create CSV file.\n";
        return 1;
    }

    csv << "Matrix_Size,Algorithm,Time,GFLOPS,Speedup\n";

    for (int N : sizes)
    {
        benchmark(N, csv);
    }

    csv.close();

    cout << "\nBenchmark completed successfully!\n";
    cout << "Results saved in results/benchmark.csv\n";

    return 0;
}
