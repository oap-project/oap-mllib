/* file: error_handling.h */
/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/*
!  Content:
!    Auxiliary error-handling functions used in C++ samples
!******************************************************************************/

#include "error_handling.h"
#include "Logger.h"
#include <iostream>

void checkAllocation(void *ptr) {
    if (!ptr) {
        logger::println(logger::ERROR, "Error: Memory allocation failed");
        exit(-1);
    }
}

void checkPtr(void *ptr) {
    if (!ptr) {
        logger::println(logger::ERROR, "Error: NULL pointer");
        exit(-2);
    }
}

void fileOpenError(const char *filename) {
    logger::println(logger::ERROR, "Unable to open file '%s'", filename);
    exit(fileError);
}

void fileReadError() {
    logger::println(logger::ERROR, "Unable to read next line");
    exit(fileError);
}

void sparceFileReadError() {
    logger::println(logger::ERROR, "Incorrect format of file");
    exit(fileError);
}

void deviceError() {
    logger::println(logger::ERROR, "Error: no supported device, please select HOST/CPU/GPU");
    exit(-1);
}
