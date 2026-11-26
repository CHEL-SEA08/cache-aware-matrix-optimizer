
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = pd.read_csv("results/benchmark.csv")

algorithms = data["Algorithm"].unique()

# Graph 1: Execution Time
plt.figure(figsize=(10, 6))

for algorithm in algorithms:
    subset = data[data["Algorithm"] == algorithm]
    plt.plot(subset["Matrix_Size"], subset["Time"],
             marker="o", label=algorithm)

plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Matrix Multiplication - Execution Time")
plt.legend()
plt.grid(True)
plt.savefig("plots/execution_time.png", dpi=300)
plt.close()

# Graph 2: GFLOPS
plt.figure(figsize=(10, 6))

for algorithm in algorithms:
    subset = data[data["Algorithm"] == algorithm]
    plt.plot(subset["Matrix_Size"], subset["GFLOPS"],
             marker="o", label=algorithm)

plt.xlabel("Matrix Size")
plt.ylabel("GFLOPS")
plt.title("Matrix Multiplication - GFLOPS Comparison")
plt.legend()
plt.grid(True)
plt.savefig("plots/gflops.png", dpi=300)
plt.close()

# Graph 3: Speedup
plt.figure(figsize=(10, 6))

for algorithm in algorithms:
    subset = data[data["Algorithm"] == algorithm]
    plt.plot(subset["Matrix_Size"], subset["Speedup"],
             marker="o", label=algorithm)

plt.xlabel("Matrix Size")
plt.ylabel("Speedup")
plt.title("Matrix Multiplication - Speedup Comparison")
plt.legend()
plt.grid(True)
plt.savefig("plots/speedup.png", dpi=300)
plt.close()

print("All three graphs generated successfully!")
print("1. plots/execution_time.png")
print("2. plots/gflops.png")
print("3. plots/speedup.png")
