# C compiler
HOST_CC = g++
HOST_CC_FLAGS = -g -O2

GEM5_CC = aarch64-linux-g++
GEM5_CC_FLAGS = -g -O2 -static 

DATA_DIR = $(HOME)/apps/hotspot/data
GEM5_DIR = $(HOME)/gem5

compile_all: hotspot_gem5 hotspot_host

compile_host: hotspot_host

compile_gem5: hotspot_gem5

run_host: hotspot_host
	@echo "======================================================================"
	@echo "WARNING: Running with 1000 iterations... "
	@echo "you may change the number of iterations and the problem size"
	@echo "======================================================================"
	./hotspot_host 1024 1024 1000 1 $(DATA_DIR)/temp_1024 $(DATA_DIR)/power_1024 output_host.out

run_gem5: hotspot_gem5
	@echo "======================================================================"
	@echo "WARNING: Running the smallest problem size (64) with 2 iterations... "
	@echo "you should change the number of iterations and the problem size"
	@echo "for the final results"
	@echo "======================================================================"
	$(GEM5_DIR)/build/ARM/gem5.opt $(GEM5_DIR)/configs/example/se.py --cpu-type=ex5_LITTLE --caches --svevl=1 -c hotspot_gem5 -o '64 64 2 1 $(DATA_DIR)/temp_64 $(DATA_DIR)/power_64 output_gem5.out'

hotspot_host: hotspot.cpp Makefile 
	$(HOST_CC) $(HOST_CC_FLAGS) hotspot.cpp -o hotspot_host

hotspot_gem5: hotspot.cpp Makefile 
	$(GEM5_CC) $(GEM5_CC_FLAGS) hotspot.cpp -o hotspot_gem5 

clean:
	rm -f hotspot_host hotspot_gem5 output_*.out
