#include <list>
#include <memory>
#include <unistd.h>

#include "GPU.h"
#include "Logger.h"
#define STORE_TIMEOUT_SEC 120
#define KVS_CREATE_SUCCESS 0
#define KVS_CREATE_FAILURE -1

typedef std::shared_ptr<sycl::queue> queuePtr;
static std::mutex g_mtx;
static std::vector<sycl::queue> g_queueVector;
std::shared_ptr<file_store> store;

static std::vector<sycl::device> get_gpus() {
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    logger::printerrln(logger::ERROR, "No GPUs!");
    exit(-1);

    return {};
}

int create_kvs_by_store(std::shared_ptr<file_store> store, int rank,
                        ccl::shared_ptr_class<ccl::kvs> &kvs) {
    logger::println(logger::INFO, "OneCCL (native): create_kvs_by_store ");
    auto t1 = std::chrono::high_resolution_clock::now();
    ccl::kvs::address_type main_addr;
    auto start = std::chrono::system_clock::now();
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        if (store->write((void *)main_addr.data(), main_addr.size()) < 0) {
            logger::println(
                logger::INFO,
                "OneCCL (native): error occurred during write attempt");
            kvs.reset();
            return KVS_CREATE_FAILURE;
        }
        auto end = std::chrono::system_clock::now();
        auto exec_time =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                .count();
        logger::println(logger::INFO,
                        "OneCCL (native): write to store time %f secs",
                        exec_time / 1000);
    } else {
        if (store->read((void *)main_addr.data(), main_addr.size()) < 0) {
            logger::println(
                logger::INFO,
                "OneCCL (native): error occurred during read attempt");
            kvs.reset();
            return KVS_CREATE_FAILURE;
        }
        auto end = std::chrono::system_clock::now();
        auto exec_time =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                .count();
        logger::println(logger::INFO,
                        "OneCCL (native): read from store time %f secs",
                        exec_time / 1000);
        kvs = ccl::create_kvs(main_addr);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    return KVS_CREATE_SUCCESS;
}

static int getLocalRank(ccl::communicator &comm, int size, int rank) {
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

sycl::queue getAssignedGPU(const ComputeDevice device, int *gpu_indices) {
    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        logger::printerrln(
            logger::ERROR,
            "Not implemented for HOST/CPU device, Please run on GPU device.");
        exit(-1);
    }
    case ComputeDevice::gpu: {
        logger::println(logger::INFO, "selector GPU");
        auto gpus = get_gpus();
        auto rank_gpu = gpus[0];
        sycl::queue q{rank_gpu};
        return q;
    }

    default: {
        logger::printerrln(logger::ERROR, "No Device!");
        exit(-1);
    }
    }
}

sycl::queue getQueue(const ComputeDevice device) {
    logger::println(logger::INFO, "Get Queue");

    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        logger::printerrln(
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
        logger::printerrln(logger::ERROR, "No Device!");
        exit(-1);
    }
    }
}

preview::spmd::communicator<preview::spmd::device_memory_access::usm>
createDalCommunicator(const jint executorNum, const jint rank,
                      const ccl::string kvs_store_path) {
    auto gpus = get_gpus();

    auto t1 = std::chrono::high_resolution_clock::now();

    ccl::init();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();

    logger::println(logger::INFO, "OneCCL singleton init took %f secs",
                    duration / 1000);

    t1 = std::chrono::high_resolution_clock::now();
    ccl::shared_ptr_class<ccl::kvs> kvs;

    store = std::make_shared<file_store>(
        kvs_store_path, rank, std::chrono::seconds(STORE_TIMEOUT_SEC));
    if (create_kvs_by_store(store, rank, kvs) != KVS_CREATE_SUCCESS) {
        logger::println(logger::INFO, "can not create kvs by store");
        throw std::runtime_error("Failed to create communicator");
    }
    t2 = std::chrono::high_resolution_clock::now();
    duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO, "OneCCL (native): create kvs took %f secs",
                    duration / 1000);
    sycl::queue queue{gpus[0]};
    t1 = std::chrono::high_resolution_clock::now();
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executorNum, rank, kvs);
    t2 = std::chrono::high_resolution_clock::now();
    duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    return comm;
}
