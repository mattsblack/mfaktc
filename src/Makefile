# configure path to CUDA Toolkit
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include/
CUDA_LIB = -L$(CUDA_DIR)/lib64/

DEBUG = 0
STATIC_RUNTIME = 0
STATIC_CUDA = 0

NVCC = nvcc

# toolkit release as a number, e.g. 1204 for CUDA 12.4 (empty if nvcc is not found)
NVCC_RELEASE := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | awk '{ printf "%d%02d", $$1, $$2 }')

# newest GCC major version the toolkit accepts as host compiler; override with
# e.g. `make NVCC_MAX_GCC=13` if your toolkit's release notes say otherwise
ifeq ($(NVCC_RELEASE),)
  NVCC_MAX_GCC ?= 15
else ifeq ($(shell [ $(NVCC_RELEASE) -ge 1300 ] && echo y),y)
  NVCC_MAX_GCC ?= 15
else ifeq ($(shell [ $(NVCC_RELEASE) -ge 1208 ] && echo y),y)
  NVCC_MAX_GCC ?= 14
else ifeq ($(shell [ $(NVCC_RELEASE) -ge 1204 ] && echo y),y)
  NVCC_MAX_GCC ?= 13
else
  NVCC_MAX_GCC ?= 12
endif

# compiler settings for host code. GNU make predefines CC=cc and CXX=g++, so
# using ?= here does not actually select a CUDA-supported compiler. Use the
# newest versioned GCC (g++-N) the toolkit supports when one is installed, and
# the unversioned one otherwise. `make CC=... CXX=...` overrides this.
HOST_GCC_SUFFIX := $(shell for v in $$(seq $(NVCC_MAX_GCC) -1 9); do \
                     if command -v g++-$$v >/dev/null 2>&1 && command -v gcc-$$v >/dev/null 2>&1; then echo -$$v; exit; fi; done)
ifeq ($(origin CC), default)
  CC := $(shell command -v gcc$(HOST_GCC_SUFFIX))
endif
ifeq ($(origin CXX), default)
  CXX := $(shell command -v g++$(HOST_GCC_SUFFIX))
  ifeq ($(shell $(CXX) --version 2>/dev/null | grep -c clang),0)
    HOST_GCC_MAJOR := $(shell $(CXX) -dumpversion 2>/dev/null | cut -d. -f1)
  endif
  ifneq ($(HOST_GCC_MAJOR),)
    ifeq ($(shell [ $(HOST_GCC_MAJOR) -gt $(NVCC_MAX_GCC) ] && echo y),y)
      $(warning WARNING: $(CXX) is GCC $(HOST_GCC_MAJOR), but this CUDA toolkit supports GCC $(NVCC_MAX_GCC) at most; install g++-$(NVCC_MAX_GCC) or set CC and CXX)
    endif
  endif
endif
CFLAGS = -Wall -Wextra $(CUDA_INCLUDE)

ifeq ($(DEBUG), 1)
  CFLAGS += -g -Og
else
  CFLAGS += -O3 -ffast-math -flto -ffunction-sections -fdata-sections
  # clang (also as "gcc" on macOS) doesn't support -funroll-all-loops
  ifeq ($(shell $(CC) --version 2>/dev/null | grep -c clang),0)
    CFLAGS_EXTRA_SIEVE = -funroll-all-loops
  endif
endif

# compiler settings for .cu files (GPU)
# nvcc otherwise searches PATH for its own host compiler, independently of CC
# and CXX. Explicitly use the same supported C++ compiler selected above.
NVCCFLAGS = $(CUDA_INCLUDE) -ccbin $(CXX) --ptxas-options=-v -Wno-deprecated-gpu-targets

# generate code for compute capabilities - see this table for supported
# versions: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
#
# A cubin is built for each compute capability in CUDA_ARCHS_WANTED that this
# nvcc can target (nvcc --list-gpu-code), so the same Makefile works from CUDA
# 12.4, which predates Blackwell, to CUDA 13.x, which dropped Maxwell, Pascal
# and Volta. Turing (7.5) is the oldest GPU both of those toolkits support.
#
# CUDA_LEGACY_ARCHS=1 also builds for Maxwell, Pascal and Volta; only CUDA 12.x
# and older can do this. `make CUDA_ARCHS="86 89"` skips detection and builds
# exactly the listed compute capabilities.
#
# note: mfaktc 0.24.x dropped support for compute capability 1.x GPUs - use the '0.23' branch for older devices
CUDA_ARCHS_WANTED = 75 80 86 87 88 89 90 100 103 110 120 121
CUDA_LEGACY_ARCHS = 0
ifeq ($(CUDA_LEGACY_ARCHS), 1)
  CUDA_ARCHS_WANTED := 50 52 53 60 61 62 70 72 $(CUDA_ARCHS_WANTED)
endif
ifeq ($(origin CUDA_ARCHS), undefined)
  NVCC_GPU_CODES := $(patsubst sm_%,%,$(filter sm_%,$(shell $(NVCC) --list-gpu-code 2>/dev/null)))
  CUDA_ARCHS := $(filter $(NVCC_GPU_CODES),$(CUDA_ARCHS_WANTED))
  ifeq ($(CUDA_ARCHS),)
    # nvcc is missing or predates --list-gpu-code
    CUDA_ARCHS := 75 80 86 89 90
  endif
endif
NVCCFLAGS += $(foreach cc,$(CUDA_ARCHS),--generate-code arch=compute_$(cc),code=sm_$(cc))

# Also embed PTX for the oldest architecture so that GPUs without a matching
# cubin can JIT-compile the kernels. The driver must be at least as new as the
# toolkit to load that PTX; mfaktc refuses to start on an older driver anyway.
CUDA_PTX_ARCH = $(firstword $(CUDA_ARCHS))
ifneq ($(CUDA_PTX_ARCH),)
  NVCCFLAGS += --generate-code arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)
endif

# pass some options to the C host compiler
ifeq ($(DEBUG), 1)
  NVCCFLAGS += -g -G -O1 --compiler-options="-Wall -Wextra -Og"
else
  NVCCFLAGS += -O3 --compiler-options="-Wall -Wextra -O3"
endif

# Linker
LD = $(CXX)

ifeq ($(STATIC_CUDA), 1)
  LDFLAGS = -flto -Wl,--gc-sections $(CUDA_LIB) -lcudart_static -lm -ldl -lrt -lpthread
else
  LDFLAGS = -flto -Wl,--gc-sections $(CUDA_LIB) -lcudart -lm -ldl -lrt -lpthread
endif

ifeq ($(STATIC_RUNTIME), 1)
  LDFLAGS += -static-libstdc++ -static-libgcc
endif

INSTALL = install

##############################################################################

CSRC  = sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c \
	filelocking.c signal_handler.c output.c crc.c
CUSRC = tf_96bit.cu tf_barrett96.cu tf_barrett96_gs.cu gpusieve.cu \
        cuda_utils.cu

COBJS  = $(CSRC:.c=.o)
CUOBJS = $(CUSRC:.cu=.o) tf_75bit.o

##############################################################################

all: ../mfaktc ../mfaktc.ini.example

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

# install the default settings as mfaktc.ini.example so that an existing
# mfaktc.ini isn't overwritten; mfaktc creates mfaktc.ini from it on first
# start unless a custom INI file is specified
../mfaktc.ini.example : mfaktc.ini
	$(INSTALL) -m 644 $< $@

.PHONY: all clean
