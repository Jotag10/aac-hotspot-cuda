HOST_CC = g++
HOST_CC_FLAGS = -g -O3

CUDA_CC = nvcc
CUDA_CC_FLAGS = -g -O3 -lm

DATA_DIR = ../data

RM=rm -f

KERNEL = kernel

OBJDIR = build
SOBJ_host = hotspot_host.o
SOBJ_cuda = hotspot_cuda.o

OBJ_host := $(patsubst %.o,$(OBJDIR)/%.o,$(SOBJ_host))
OBJ_cuda := $(patsubst %.o,$(OBJDIR)/%.o,$(SOBJ_cuda))


compile_all: hotspot_host hotspot_cuda

compile_host: hotspot_host

compile_cuda: hotspot_cuda


link_objs_host = $(HOST_CC) $(HOST_CC_FLAGS) -o $@ $(OBJ_host) $(OBJDIR)/$(KERNEL)_$@.o $(LIBS)
link_objs_cuda = $(CUDA_CC) $(CUDA_CC_FLAGS) -o $@ $(OBJ_cuda) $(OBJDIR)/$(KERNEL)_$@.o $(LIBS)

run_host: hotspot_host
	@echo "======================================================================"
	@echo "Host code"
	@echo "problem size (1024) with 1000 iterations... "
	@echo "======================================================================"
	./hotspot_host 1024 1024 1 1000 $(DATA_DIR)/temp_1024 $(DATA_DIR)/power_1024 output_host.out
	
run_cuda: hotspot_cuda
	@echo "======================================================================"
	@echo "Cuda code"
	@echo "problem size (1024) with 1000 iterations... "
	@echo "======================================================================"
	./hotspot_cuda 1024 1024 1 1000 $(DATA_DIR)/temp_1024 $(DATA_DIR)/power_1024 output_host.out

hotspot_host: $(OBJ_host)
	$(link_objs_host)

hotspot_cuda: $(OBJ_cuda) $(KERNEL).cu
	$(CUDA_CC) $(CUDA_CC_FLAGS) -o $(OBJDIR)/$(KERNEL)_$@.o -c $(KERNEL).cu 
	$(link_objs_cuda)

$(OBJDIR)/$(SOBJ_host) : hotspot.cpp
	$(HOST_CC) $(HOST_CC_FLAGS) -c $< -o $@ -DHOST
	
$(OBJDIR)/$(SOBJ_cuda) : hotspot.cpp
	$(CUDA_CC) $(CUDA_CC_FLAGS) -c $< -o $@

clean :

	$(RM) hotspot_host
	$(RM) hotspot_cuda
	$(RM) -r build/

$(shell mkdir -p $(OBJDIR))
