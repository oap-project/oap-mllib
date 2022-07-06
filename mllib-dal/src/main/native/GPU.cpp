#include <list>
#include <memory>
#include <unistd.h>

#include "GPU.h"

typedef std::shared_ptr<sycl::queue> queuePtr;

static std::mutex mtx;
static std::vector<sycl::queue> cVector;

static std::vector<sycl::device> get_gpus() {
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    std::cout << "No GPUs!" << std::endl;
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

sycl::device getAssignedGPU(ccl::communicator &comm, int size, int rankId,
                            jint *gpu_indices, int n_gpu) {
    auto local_rank = getLocalRank(comm, size, rankId);
    auto gpus = get_gpus();

    std::cout << "rank: " << rankId << " size: " << size
              << " local_rank: " << local_rank << " n_gpu: " << n_gpu
              << std::endl;

    auto gpu_selected = gpu_indices[local_rank % n_gpu];
    std::cout << "GPU selected for current rank: " << gpu_selected << std::endl;

    // In case gpu_selected index is larger than number of GPU SYCL devices
    auto rank_gpu = gpus[gpu_selected % gpus.size()];

    return rank_gpu;
}

static sycl::queue getSyclQueue(const sycl::device device) {
    mtx.lock();
    if (!cVector.empty()) {
        const auto device = cVector[0];
        mtx.unlock();
        return device;
    } else {
        sycl::queue queue{device};
        cVector.push_back(queue);
        const auto device = cVector[0];
        mtx.unlock();
        return device;
    }
}

sycl::queue getQueue(const compute_device device) {
    std::cout << "Get Queue" << std::endl;

    switch (device) {
    case compute_device::host:
    case compute_device::cpu: {
        std::cout
            << "Not implemented for HOST/CPU device, Please run on GPU device."
            << std::endl;
        exit(-1);
    }
    case compute_device::gpu: {
        std::cout << "selector GPU" << std::endl;
        auto device_gpu = sycl::gpu_selector{}.select_device();
        return getSyclQueue(device_gpu);
    }

    default: {
        std::cout << "No Device!" << std::endl;
        exit(-1);
    }
    }
}
