# configure path to CUDA Toolkit
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include/
CUDA_LIB = -L$(CUDA_DIR)/lib64/

DEBUG = 0

# compiler settings for host code. GNU make predefines CC=cc and CXX=g++, so
# using ?= here does not actually select a CUDA-supported compiler. Prefer GCC
# 15 when it is installed, while still allowing `make CC=... CXX=...` overrides.
ifeq ($(origin CC), default)
  CC := $(shell command -v gcc-15 2>/dev/null || command -v gcc)
endif
ifeq ($(origin CXX), default)
  CXX := $(shell command -v g++-15 2>/dev/null || command -v g++)
endif
CFLAGS = -Wall -Wextra $(CUDA_INCLUDE)

ifeq ($(DEBUG), 1)
  CFLAGS += -g -Og
else
  CFLAGS += -O3 -ffast-math -flto -ffunction-sections -fdata-sections
  CFLAGS_EXTRA_SIEVE = -funroll-all-loops
endif

# compiler settings for .cu files (GPU)
NVCC = nvcc
# nvcc otherwise searches PATH for its own host compiler, independently of CC
# and CXX. Explicitly use the same supported C++ compiler selected above.
NVCCFLAGS = $(CUDA_INCLUDE) -ccbin $(CXX) --ptxas-options=-v -Wno-deprecated-gpu-targets

# generate code for compute capabilities - see this table for supported
# versions: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

# older deprecated versions - uncomment to enable
#
# note: mfaktc 0.24.x dropped support for compute capability 1.x GPUs - use the '0.23' branch for older devices
# NVCCFLAGS += --generate-code arch=compute_20,code=sm_20 # Fermi GPUs
# NVCCFLAGS += --generate-code arch=compute_30,code=sm_30 # early Kepler GPUs
# NVCCFLAGS += --generate-code arch=compute_35,code=sm_35 # later Kepler GPUs; funnel shift added in 3.2

# removed in CUDA 13.x - uncomment to enable
# NVCCFLAGS += --generate-code arch=compute_50,code=sm_50 # Maxwell GPUs
# NVCCFLAGS += --generate-code arch=compute_60,code=sm_60 # Pascal GPUs
# NVCCFLAGS += --generate-code arch=compute_61,code=sm_61
# NVCCFLAGS += --generate-code arch=compute_62,code=sm_62
# NVCCFLAGS += --generate-code arch=compute_70,code=sm_70 # Volta GPUs
# NVCCFLAGS += --generate-code arch=compute_72,code=sm_72

# currently supported
NVCCFLAGS += --generate-code arch=compute_75,code=sm_75 # Turing GPUs; slightly faster than 7.0
NVCCFLAGS += --generate-code arch=compute_80,code=sm_80 # Ampere GPUs
NVCCFLAGS += --generate-code arch=compute_86,code=sm_86
NVCCFLAGS += --generate-code arch=compute_87,code=sm_87
NVCCFLAGS += --generate-code arch=compute_88,code=sm_88
NVCCFLAGS += --generate-code arch=compute_89,code=sm_89 # Ada Lovelace GPUs
NVCCFLAGS += --generate-code arch=compute_90,code=sm_90 # Hopper GPUs
NVCCFLAGS += --generate-code arch=compute_100,code=sm_100 # Blackwell data-center GPUs
NVCCFLAGS += --generate-code arch=compute_103,code=sm_103
NVCCFLAGS += --generate-code arch=compute_110,code=sm_110
NVCCFLAGS += --generate-code arch=compute_120,code=sm_120 # Blackwell consumer GPUs
NVCCFLAGS += --generate-code arch=compute_121,code=sm_121

# Keep forward-compatible PTX in the fat binary for CUDA-capable devices that
# do not have an exact cubin above.  CUDA 13.2.1 supports JIT-compiling this
# baseline PTX for every architecture supported by mfaktc.
NVCCFLAGS += --generate-code arch=compute_75,code=compute_75

# pass some options to the C host compiler
ifeq ($(DEBUG), 1)
  NVCCFLAGS += -g -G -O1 --compiler-options="-Wall -Wextra -Og"
else
  NVCCFLAGS += -O3 --compiler-options="-Wall -Wextra -O3"
endif

# Linker
LD = $(CXX)
LDFLAGS = -flto -Wl,--gc-sections $(CUDA_LIB) -lcudart_static -lm -ldl -lrt -lpthread

INSTALL = install

##############################################################################

CSRC  = sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c \
	filelocking.c signal_handler.c output.c crc.c
CUSRC = tf_96bit.cu tf_barrett96.cu tf_barrett96_gs.cu gpusieve.cu \
        cuda_utils.cu

COBJS  = $(CSRC:.c=.o)
CUOBJS = $(CUSRC:.cu=.o) tf_75bit.o

##############################################################################

all: ../mfaktc ../mfaktc.ini

../mfaktc : $(COBJS) $(CUOBJS)
	$(LD) $^ -o $@ $(LDFLAGS)

clean :
	rm -f *.o *~

sieve.o : sieve.c
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA_SIEVE) -c $< -o $@

tf_75bit.o : tf_96bit.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -DSHORTCUT_75BIT

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

../%.ini : %.ini
	$(INSTALL) -m 644 $< ..

.PHONY: all clean
