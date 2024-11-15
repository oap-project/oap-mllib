#include <list>
#include <memory>
#include <unistd.h>

#include "GPU.h"
#include "Logger.h"

typedef std::shared_ptr<sycl::queue> queuePtr;

static std::mutex g_mtx;
static std::vector<sycl::queue> g_queueVector;

std::vector<sycl::device> get_gpus() {
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
