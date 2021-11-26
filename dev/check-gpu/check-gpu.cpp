#include <CL/sycl.hpp>

int check_gpu() {    
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

int main (int argc, char * argv[]) {
    return check_gpu();        
}