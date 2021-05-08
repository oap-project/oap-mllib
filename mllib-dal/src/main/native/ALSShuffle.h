/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#pragma once

#include <jni.h>
#include <oneapi/ccl.hpp>

struct Rating {
    jlong user;
    jlong item;
    jfloat rating;
} __attribute__((packed));

const int RATING_SIZE = sizeof(Rating);

typedef std::vector<unsigned char> ByteBuffer;
typedef std::vector<Rating> RatingPartition;

jlong getPartiton(jlong key, jlong totalKeys, long nBlocks);
Rating *shuffle_all2all(ccl::communicator &comm,
                        std::vector<RatingPartition> &partitions,
                        size_t nBlocks, size_t &ratingsNum, size_t &csrRowNum);
