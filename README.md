# Cache-Aware Matrix Multiplication Optimizer

## Project Overview

The Cache-Aware Matrix Multiplication Optimizer is a high-performance computing project that explores different techniques to optimize matrix multiplication using C++.

The project implements and compares five approaches: Naive Matrix Multiplication, Loop Reordering, Cache Blocking, AVX2 SIMD Vectorization, and OpenMP Parallelization.

The objective is to study how memory access patterns, CPU vector instructions, and parallel execution affect computational performance.

## Technologies Used

- C++
- AVX2 and FMA
- OpenMP
- Linux (Ubuntu / WSL2)
- GCC Compiler
- Linux perf
- Python
- Pandas
- Matplotlib

## Implemented Algorithms

1. **Naive Matrix Multiplication:** Basic triple-loop implementation.
2. **Loop Reordering:** Changes loop order to improve memory access locality.
3. **Cache Blocking:** Divides matrices into smaller blocks to improve cache utilization.
4. **AVX2 SIMD:** Uses vector instructions to process multiple values simultaneously.
5. **OpenMP:** Uses parallel execution to distribute computation across CPU threads.

## Project Structure

```text
cache-aware-matrix-optimizer/
├── src/
│   └── matrix.cpp
├── scripts/
│   └── plot_results.py
├── results/
│   └── benchmark.csv
├── plots/
│   ├── execution_time.png
│   ├── gflops.png
│   └── speedup.png
├── README.md
└── matrix
```

## Compilation

Compile the C++ program using:

```bash
g++ -O2 -mavx2 -mfma -fopenmp src/matrix.cpp -o matrix
```

## Execution

```bash
./matrix
```

## Benchmarking

The program evaluates matrix sizes:

- 128 × 128
- 256 × 256
- 512 × 512
- 1024 × 1024

The benchmark records:

- Execution Time
- GFLOPS
- Speedup

Results are stored in `results/benchmark.csv`.

## Performance Visualization

Three graphs are generated using Python and Matplotlib:

1. Execution Time Comparison
2. GFLOPS Comparison
3. Speedup Comparison

Run the visualization script from the project root:

```bash
python3 scripts/plot_results.py
```

## Linux Performance Analysis

Linux `perf` was used to collect CPU performance statistics and cache-related hardware counters.

Commands:

```bash
perf stat ./matrix
```

```bash
perf stat -e cache-references,cache-misses ./matrix
```

## Key Results

In the recorded 1024 × 1024 benchmark run:

| Algorithm | Execution Time (seconds) | Speedup |
|---|---:|---:|
| Naive | 12.844676 | 1.00× |
| Loop Reordered | 0.519402 | 24.73× |
| Cache Blocked | 0.546605 | 23.50× |
| AVX2 SIMD | 0.245146 | 52.40× |
| OpenMP | 0.130648 | 98.32× |

OpenMP achieved the highest recorded performance for the largest tested matrix.

## Conclusion

This project demonstrates how algorithmic optimization, cache-aware memory access, SIMD vectorization, and multithreading can improve matrix multiplication performance.

The experiments show that different optimization techniques provide different benefits depending on matrix size and system configuration.

## Future Enhancements

- Repeated benchmark runs with statistical summaries.
- Testing additional cache block sizes.
- Comparing different compiler optimization levels.
- More detailed hardware performance analysis.
- Testing larger matrix dimensions.
