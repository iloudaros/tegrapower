import torch
import time

def benchmark_matmul(name, func, runs=9999999999):
    """A simple benchmarking function with CUDA synchronization."""
    # Warm-up runs to let the GPU stabilize its clocks and cache
    for _ in range(10):
        func()

    # Ensure all previous GPU work is done before starting the timer
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(runs):
        func()
    # Wait for the benchmarked work to complete
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / runs
    print(f"[{name}] Average time: {avg_time_ms:.4f} ms")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available on this PyTorch build. Exiting.")
        exit()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print("-" * 40)

    device = torch.device("cuda")
    dtype = torch.float32

    # === 1. Standard GEMM Test (A @ B) ===
    print(f"Testing Standard GEMM with {dtype}")
    #M, K, N = 3072, 2048, 1024
    M, K, N = 3072, 1024, 512

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    benchmark_matmul("Standard GEMM", lambda: torch.matmul(a, b))

    print("-" * 40)

    """
    # === 2. Batched GEMM (BMM) Test (Batch @ Batch) ===
    print(f"Testing Batched GEMM (torch.bmm) with {dtype}")
    # 3072 2048 1024
    Batch, M, K, N = 1, 3072, 2048, 1024

    a_batch = torch.randn(Batch, M, K, device=device, dtype=dtype)
    b_batch = torch.randn(Batch, K, N, device=device, dtype=dtype)
    benchmark_matmul("Batched GEMM (BMM)", lambda: torch.bmm(a_batch, b_batch))

    print("-" * 40)
    
    # === 3. Batched GEMM with Reused RHS (Broadcasting) ===
    # This is the pattern for applying a Linear layer to a sequence
    print(f"Testing Batched GEMM with Reused RHS (Broadcasting) with {dtype}")
    Batch, SeqLen, InFeat, OutFeat = 32, 128, 512, 1024
    
    # Input batch (e.g., batch_size x sequence_length x embedding_dim)
    lhs_batch = torch.randn(Batch * SeqLen, InFeat, device=device, dtype=dtype).view(Batch, SeqLen, InFeat)
    # Reused matrix (e.g., a Linear layer's weight)
    rhs_reused = torch.randn(InFeat, OutFeat, device=device, dtype=dtype)
    
    print(f"Shapes: LHS={lhs_batch.shape}, RHS={rhs_reused.shape}")
    benchmark_matmul("Broadcast GEMM", lambda: lhs_batch @ rhs_reused)
    
    # Verify the output shape
    result = lhs_batch @ rhs_reused
    print(f"Resulting shape: {result.shape}")
    print("-" * 40)
    """

