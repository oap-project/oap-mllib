# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use gcc for CPU and dpcpp for GPU
ifeq ($(PLATFORM_PROFILE),CPU_ONLY_PROFILE)
  CC := gcc
  CXX := g++
else ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  CC := dpcpp
  CXX := dpcpp
endif

RM := rm -rf

PLATFORM_PROFILE ?= CPU_ONLY_PROFILE

$(info )
$(info === Profile is $(PLATFORM_PROFILE) ===)
$(info )

CFLAGS_COMMON := -g -Wall -Wno-deprecated-declarations -fPIC -std=c++17 \
                 -I$(CCL_ROOT)/include/cpu_gpu_dpcpp/oneapi/ \
                 -I$(I_MPI_ROOT)/include \
                 -I$(DAALROOT)/include \
                 -I$(CMPLR_ROOT)/linux/include \
                 -I$(CMPLR_ROOT)/linux/include/sycl

ifeq ($(PLATFORM_PROFILE),CPU_ONLY_PROFILE)
  CFLAGS := $(CFLAGS_COMMON)
else ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  CFLAGS := $(CFLAGS_COMMON) -fsycl
else
  $(error Unknow building profile, should be CPU_ONLY_PROFILE or CPU_GPU_PROFILE)
  exit 1
endif

INCS := -I $(JAVA_HOME)/include \
        -I $(JAVA_HOME)/include/linux \
        -I $(CCL_ROOT)/include/cpu_gpu_dpcpp \
        -I $(DAALROOT)/include \
        -I /opt/rh/devtoolset-7/root/usr/include/c++/7 \
        -I /opt/rh/devtoolset-7/root/usr/include/c++/7/x86_64-redhat-linux \
        -I ./javah \
        -I ./

# Use static link if possible, TBB is only available as dynamic libs
LIBS_COMMON := -L$(CCL_ROOT)/lib/cpu_gpu_dpcpp -lccl \
        -L$(CMPLR_ROOT)/linux/compiler/lib/intel64_lin -l:libirc.a \
        -L$(DAALROOT)/lib/intel64 -l:libonedal_core.a -l:libonedal_thread.a -l:libonedal_dpc.a \
        -L$(TBBROOT)/lib/intel64/gcc4.8 -ltbb -ltbbmalloc \
        -L$(I_MPI_ROOT)

# ONEDAL_LIBS := -lonedal_dpc \
#                -lonedal_core \
#                -lonedal_thread
ONEDAL_LIBS := "$(DAALROOT)"/lib/intel64/libonedal_dpc.a \
               "$(DAALROOT)"/lib/intel64/libonedal_core.a \
               "$(DAALROOT)"/lib/intel64/libonedal_thread.a

ifeq ($(PLATFORM_PROFILE),CPU_ONLY_PROFILE)
  LIBS := $(LIBS_COMMON) $(ONEDAL_LIBS)
else ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  LIBS := $(LIBS_COMMON) $(ONEDAL_LIBS) -l:libonedal_sycl.a
endif

CPP_SRCS += \
  ./OneCCL.cpp ./OneDAL.cpp ./service.cpp ./error_handling.cpp \
  ./KMeansOneAPIImpl.cpp \
  ./PCAOneAPIImpl.cpp \
  ./ALSDALImpl.cpp ./ALSShuffle.cpp \
  ./NaiveBayesDALImpl.cpp \
  ./LinearRegressionDALImpl.cpp \
  ./CorrelationOneAPIImpl.cpp \
  ./SummarizerDALImpl.cpp \
  ./oneapi/dal/HomogenTableImpl.cpp \
  ./oneapi/dal/SimpleMetadataImpl.cpp \
  ./oneapi/dal/ColumnAccessorImpl.cpp \
  ./oneapi/dal/RowAccessorImpl.cpp


OBJS += \
  ./OneCCL.o ./OneDAL.o ./service.o ./error_handling.o \
  ./KMeansOneAPIImpl.o \
  ./PCAOneAPIImpl.o \
  ./ALSDALImpl.o ./ALSShuffle.o \
  ./NaiveBayesDALImpl.o \
  ./LinearRegressionDALImpl.o \
  ./CorrelationOneAPIImpl.o \
  ./SummarizerDALImpl.o \
  ./oneapi/dal/HomogenTableImpl.o \
  ./oneapi/dal/SimpleMetadataImpl.o \
  ./oneapi/dal/ColumnAccessorImpl.o  \
  ./oneapi/dal/RowAccessorImpl.o

DEFINES=-D$(PLATFORM_PROFILE)


ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  CPP_SRCS += ./GPU.cpp
  OBJS += ./GPU.o
endif

# Output Binary
OUTPUT = ../../../src/main/resources/lib/libMLlibDAL.so


all: $(OUTPUT)

# Compile
%.o: %.cpp
	@echo 'Building file: $<'
	$(CXX) $(CFLAGS) $(INCS) $(DEFINES) -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

# Link
$(OUTPUT): $(OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Linker'
	$(CXX) $(CFLAGS) -shared -o $(OUTPUT) $(OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	@echo 'Cleaning up'
	-$(RM) $(OBJS) $(OUTPUT)
	-@echo ' '

.PHONY: all clean
