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
  CC := icpx
  CXX := icpx
endif

RM := rm -rf

PLATFORM_PROFILE ?= CPU_ONLY_PROFILE

$(info )
$(info === Profile is $(PLATFORM_PROFILE) ===)
$(info )

CFLAGS_COMMON := -Wall -Wno-deprecated-declarations -fPIC -std=c++17 \
                 -I $(I_MPI_ROOT)/include \
                 -I $(DAALROOT)/include \
                 -I $(CCL_ROOT)/include/cpu/oneapi/ \
                 -I $(CMPLR_ROOT)/linux/include \
                 -I $(CMPLR_ROOT)/linux/include/sycl

ifeq ($(PLATFORM_PROFILE),CPU_ONLY_PROFILE)
  CFLAGS := $(CFLAGS_COMMON)
else ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  CFLAGS := $(CFLAGS_COMMON) -fsycl \
            -fsycl-device-code-split=per_kernel \
            -I $(CCL_ROOT)/include/cpu_gpu_dpcpp/oneapi/
else
  $(error Unknow building profile, should be CPU_ONLY_PROFILE or CPU_GPU_PROFILE)
  exit 1
endif

INCS := -I $(CCL_ROOT)/include/cpu \
        -I $(JAVA_HOME)/include \
        -I $(JAVA_HOME)/include/linux \
        -I $(DAALROOT)/include \
        -I ./javah \
        -I ./

ifeq ($(PLATFORM_PROFILE),CPU_ONLY_PROFILE)
  INCS := $(INCS)
else ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  INCS := $(INCS) -I $(CCL_ROOT)/include/cpu_gpu_dpcpp
else
  $(error Unknow building profile, should be CPU_ONLY_PROFILE or CPU_GPU_PROFILE)
  exit 1
endif

# Use static link if possible, TBB is only available as dynamic libs
LIBS_COMMON := -L$(CCL_ROOT)/lib/cpu -lccl \
        -L$(CMPLR_ROOT)/linux/compiler/lib/intel64_lin -l:libirc.a \
        -L$(DAALROOT)/lib/intel64 -lonedal_core -lonedal_thread -lonedal_dpc \
        -L$(TBBROOT)/lib/intel64/gcc4.8 -ltbb -ltbbmalloc \
        -L$(I_MPI_ROOT)

ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
      LIBS_COMMON := $(LIBS_COMMON) \
                     -L$(CCL_ROOT)/lib/cpu_gpu_dpcpp -lccl
endif

ifeq ($(PLATFORM_PROFILE),CPU_ONLY_PROFILE)
  LIBS := $(LIBS_COMMON) $(ONEDAL_LIBS)
else ifeq ($(PLATFORM_PROFILE),CPU_GPU_PROFILE)
  LIBS := $(LIBS_COMMON) $(ONEDAL_LIBS) -l:libonedal_sycl.a
endif

CPP_SRCS += \
  ./service.cpp ./error_handling.cpp \
  ./daal/csr_numeric_table_impl.cpp \
  ./daal/homogen_numeric_table_byte_buffer_impl.cpp \
  ./daal/merged_numeric_table_impl.cpp \
  ./daal/numeric_table_impl.cpp \
  ./daal/row_merged_numeric_table_impl.cpp \
  ./daal/data_dictionary.cpp \
  ./daal/data_feature.cpp \
  ./oneapi/dal/HomogenTableImpl.cpp \
  ./oneapi/dal/SimpleMetadataImpl.cpp \
  ./oneapi/dal/ColumnAccessorImpl.cpp \
  ./oneapi/dal/RowAccessorImpl.cpp \
  ./OneCCL.cpp ./OneDAL.cpp \
  ./KMeansImpl.cpp \
  ./PCAImpl.cpp \
  ./ALSDALImpl.cpp ./ALSShuffle.cpp \
  ./NaiveBayesDALImpl.cpp \
  ./LinearRegressionImpl.cpp \
  ./CorrelationImpl.cpp \
  ./SummarizerImpl.cpp \
  ./DecisionForestClassifierImpl.cpp \
  ./DecisionForestRegressorImpl.cpp



OBJS += \
  ./service.o ./error_handling.o \
  ./daal/csr_numeric_table_impl.o \
  ./daal/homogen_numeric_table_byte_buffer_impl.o \
  ./daal/merged_numeric_table_impl.o \
  ./daal/numeric_table_impl.o \
  ./daal/row_merged_numeric_table_impl.o \
  ./daal/data_dictionary.o \
  ./daal/data_feature.o \
  ./oneapi/dal/HomogenTableImpl.o \
  ./oneapi/dal/SimpleMetadataImpl.o \
  ./oneapi/dal/ColumnAccessorImpl.o \
  ./oneapi/dal/RowAccessorImpl.o \
  ./OneCCL.o ./OneDAL.o \
  ./KMeansImpl.o \
  ./PCAImpl.o \
  ./ALSDALImpl.o ./ALSShuffle.o \
  ./NaiveBayesDALImpl.o \
  ./LinearRegressionImpl.o \
  ./CorrelationImpl.o \
  ./SummarizerImpl.o \
  ./DecisionForestClassifierImpl.o \
  ./DecisionForestRegressorImpl.o

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
