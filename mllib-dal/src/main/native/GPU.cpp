#include <list>
#include <memory>
#include <unistd.h>

#include "GPU.h"
#include "Logger.h"

typedef std::shared_ptr<sycl::queue> queuePtr;

static std::mutex g_mtx;
static std::vector<sycl::queue> g_queueVector;

static std::vector<sycl::device> get_gpus() {
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    logger::println(logger::ERROR, "No GPUs!");
    exit(-1);

    return {};
}

static int getLocalRank(ccl::communicator &comm, int size, int rank) {
    const int MPI_MAX_PROCESSOR_NAME = 128;
    /* Obtain local rank among nodes sharing the same host name */
    char zero = static_cast<char>(0);
    std::vector<char> name(MPI_MAX_PROCESSOR_NAME + 1, zero);
    // int resultlen = 0;
    // MPI_Get_processor_name(name.data(), &resultlen);
    gethostname(name.data(), MPI_MAX_PROCESSOR_NAME);
    std::string str(name.begin(), name.end());
    std::vector<char> allNames((MPI_MAX_PROCESSOR_NAME + 1) * size, zero);
    std::vector<size_t> aReceiveCount(size, MPI_MAX_PROCESSOR_NAME + 1);
    ccl::allgatherv((int8_t *)name.data(), name.size(),
                    (int8_t *)allNames.data(), aReceiveCount, comm)
        .wait();
    int localRank = 0;
    for (int i = 0; i < rank; i++) {
        auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
        std::string nbrName(nameBegin,
                            nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
        if (nbrName == str)
            localRank++;
    }
    return localRank;

    //    return 0;
}

static sycl::queue getSyclQueue(const sycl::device device) {
    g_mtx.lock();
    if (!g_queueVector.empty()) {
        const auto device = g_queueVector[0];
        g_mtx.unlock();
        return device;
    } else {
        sycl::queue queue{device};
        g_queueVector.push_back(queue);
        const auto device = g_queueVector[0];
        g_mtx.unlock();
        return device;
    }
}

sycl::queue getAssignedGPU(const ComputeDevice device, ccl::communicator &comm,
                           int size, int rankId, jint *gpu_indices, int n_gpu) {
    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        logger::println(
            logger::ERROR,
            "Not implemented for HOST/CPU device, Please run on GPU device.");
        exit(-1);
    }
    case ComputeDevice::gpu: {
        logger::println(logger::INFO, "selector GPU");
        auto local_rank = getLocalRank(comm, size, rankId);
        auto gpus = get_gpus();

        logger::println(logger::INFO,
                        "rank: %d size: %d local_rank: %d n_gpu: %d", rankId,
                        size, local_rank, n_gpu);

        auto gpu_selected = gpu_indices[local_rank % n_gpu];
        logger::println(logger::INFO, "GPU selected for current rank: %d",
                        gpu_selected);

        // In case gpu_selected index is larger than number of GPU SYCL devices
        auto rank_gpu = gpus[gpu_selected % gpus.size()];
        sycl::queue q{rank_gpu};
        return q;
    }

    default: {
        logger::println(logger::ERROR, "No Device!");
        exit(-1);
    }
    }
}

sycl::queue getQueue(const ComputeDevice device) {
    logger::println(logger::INFO, "Get Queue");

    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        logger::println(
            logger::ERROR,
            "Not implemented for HOST/CPU device, Please run on GPU device.");
        exit(-1);
    }
    case ComputeDevice::gpu: {
        logger::println(logger::INFO, "selector GPU");
        auto device_gpu = sycl::gpu_selector{}.select_device();
        logger::println(logger::INFO, "selector GPU end");
        return getSyclQueue(device_gpu);
    }
    default: {
        logger::println(logger::ERROR, "No Device!");
        exit(-1);
    }
    }
}
