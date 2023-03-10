#include <CL/sycl.hpp>
#include<iostream>
#include<cstring>

int check_gpu() {
    std::cout << "Listing GPU devices ..." << std::endl;
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
    std::cout << "Listing CPU devices ..." << std::endl;
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
    if (argc == 1) {
        std::cout << "This program is used to list the device of the server" << std::endl <<
              "Options:" << std::endl <<
              "cpu     Check cpu device" << std::endl <<
              "gpu     Check gpu device" << std::endl;
    } else {
        char* hostname = std::getenv("HOSTNAME");
        char* device = argv[1];
        int length = strlen(device);
        for (int i = 0; i < length; i++) {
            // convert str[i] to uppercase
            device[i] = std::toupper(device[i]);
        }
        if (strcmp(device, "GPU") == 0) {
            return check_gpu();
        } else {
            return check_cpu();
        }
    }
}
