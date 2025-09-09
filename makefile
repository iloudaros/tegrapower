include .env

# Configurable workspace path (host side)
# Override by:
#   - environment variable: export WORKSPACE=/path/to/your/workspace
#   - or at invocation: make docker WORKSPACE=/path/to/your/workspace
WORKSPACE ?= $(shell realpath $(CURDIR))


# NVIDIA container image to use
# Override by:
#   - environment variable: export CONTAINER=your_preferred_image
#   - or at invocation: make docker CONTAINER=your_preferred_image
#   - setting the variable in the .env file
CONTAINER ?= nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Configurable paths for the combine script
RESULTS_DIR ?= results
OUTPUT_FILE ?= results/combined_results.csv

# Read the Jetson model from device tree
model_val := $(shell tr -d '\0' < /proc/device-tree/model)

# --- PHONY TARGETS ---
# These targets are commands, not files.
.PHONY: model jetpack_version update docker combine clean

# --- TARGETS ---

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

# Force pull the latest version of the repo
update:
	git reset --hard
	git pull

# Launch the NVIDIA container with GPU access,
# mounting your configurable workspace and the tegrastats binary
docker:
	docker run -it --rm --gpus all --runtime nvidia --network host \
		-v $(WORKSPACE):/workspace \
		-v /usr/bin/tegrastats:/usr/bin/tegrastats \
		$(CONTAINER)

# New target to combine results from all devices
combine:
	@echo "Combining results into $(OUTPUT_FILE)..."
	python3 scripts/combine_results.py --results-dir $(RESULTS_DIR) --output-file $(OUTPUT_FILE)

# Updated clean target to remove the newly generated files
clean:
	sudo rm -rf powerlogs energy_results.csv model_benchmarks.csv __pycache__ bench_with_energy.csv $(OUTPUT_FILE)

