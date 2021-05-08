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
