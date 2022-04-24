#include <CL/sycl.hpp>
#include<iostream>
#include<cstring>

int check_gpu() {
    std::cout << "Checkout GPU!" << std::endl;
    bool gpu_found = false;

    auto platforms = sycl::platform::get_platforms();
    for (auto &platform : platforms) {
        auto devices = platform.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            gpu_found = true;
            std::cout << "Platform: "
                      << platform.get_info<sycl::info::platform::name>()
                      << std::endl;
            for (auto &device : devices) {
                std::cout << "-- GPU device: "
                          << device.get_info<sycl::info::device::name>() << std::endl;
            }            
        }
    }

    if (!gpu_found)
        std::cout << "No GPU found!" << std::endl; 

    return gpu_found ? 0 : -1;
}

int check_cpu() {
    std::cout << "Checkout CPU!" << std::endl;
    bool cpu_found = false;

    auto platforms = sycl::platform::get_platforms();
    for (auto &platform : platforms) {
        auto devices = platform.get_devices(sycl::info::device_type::cpu);
        if (!devices.empty()) {
            cpu_found = true;
            std::cout << "Platform: "
                      << platform.get_info<sycl::info::platform::name>()
                      << std::endl;
            for (auto &device : devices) {
                std::cout << "-- CPU device: "
                          << device.get_info<sycl::info::device::name>() << std::endl;
            }
        }
    }

    if (!cpu_found)
        std::cout << "No CPU found!" << std::endl;

    return cpu_found ? 0 : -1;
}

int main (int argc, char * argv[]) {
    char* is_gpu = argv[1];
    std::cout << "is_gpu = " << is_gpu <<std::endl;
    if (*is_gpu == true) {
        return check_gpu();
    } else {
        return check_cpu();
    }
}