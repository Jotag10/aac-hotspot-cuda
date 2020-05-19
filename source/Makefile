# C compiler
HOST_CC = g++
HOST_CC_FLAGS = -O3

CUDA_CC = nvcc
CUDA_CC_FLAGS = -O3 -lm

DATA_DIR = $(HOME)/apps/hotspot/data

compile_all: hotspot_cuda hotspot_host

compile_host: hotspot_host

compile_cuda: hotspot_cuda

run_host: hotspot_host
	@echo "======================================================================"
	@echo "WARNING: Running with 1000 iterations... "
	@echo "you may change the number of iterations and the problem size"
	@echo "======================================================================"
	./hotspot_host 1024 1024 1000 1 $(DATA_DIR)/temp_1024 $(DATA_DIR)/power_1024 output_host.out

run_cuda: hotspot_cuda
	@echo "======================================================================"
	@echo "WARNING: Running the smallest problem size (64) with 2 iterations... "
	@echo "you should change the number of iterations and the problem size"
	@echo "for the final results"
	@echo "======================================================================"
	./hotspot_cuda 1024 1024 1000 1 $(DATA_DIR)/temp_1024 $(DATA_DIR)/power_1024 output_cuda.out

hotspot_host: hotspot.cpp Makefile 
	$(HOST_CC) $(HOST_CC_FLAGS) hotspot.cpp -o hotspot_host

hotspot_cuda: hotspot.cu Makefile 
	$(CUDA_CC) $(CUDA_CC_FLAGS) hotspot.cu -o hotspot_cuda 

clean:
	rm -f hotspot_host hotspot_cuda output_*.out
