# Configurable workspace path (host side)
# Override by:
#   - environment variable: export WORKSPACE=/path/to/your/workspace
#   - or at invocation: make docker WORKSPACE=/path/to/your/workspace
WORKSPACE ?= /home/iloudaros/pytorch_eval

# Read the Jetson model from device tree
model_val := $(shell tr -d '\0' < /proc/device-tree/model)

# Print detected Jetson model
model:
	@echo $(model_val)

# Show JetPack package information and L4T release details
jetpack_version:
	sudo apt-cache show nvidia-jetpack
	cat /etc/nv_tegra_release
	@echo ""
	@echo "Remember to check the Linux version mapping at:"
	@echo "https://docs.nvidia.com/jetson/archives/index.html"

# Launch the NVIDIA L4T ML container (R36.2.0) with GPU access,
# mounting your configurable workspace and the tegrastats binary
docker:
	docker run -it --rm --gpus all --runtime nvidia --network host \
		-v $(WORKSPACE):/workspace \
		-v /usr/bin/tegrastats:/usr/bin/tegrastats \
		nvcr.io/nvidia/l4t-ml:r36.2.0-py3
