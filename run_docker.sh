docker run --rm -it --gpus '"device=0"' \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    python3 /workspace/gemm_server_gpu.py