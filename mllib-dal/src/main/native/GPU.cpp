#include "GPU.h"

#include <list>
#include <memory>
#include <unistd.h>

#include <CL/cl.h>
#include <CL/sycl.hpp>

#include <daal_sycl.h>

#include "service.h"

std::vector<sycl::device> get_gpus()
{
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms)
    {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty())
        {
            return devices;
        }
    }
    std::cout << "No GPUs!" << std::endl;
    exit(-1);

    return {};
}

int getLocalRank(ccl::communicator & comm, int size, int rank)
{
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
    ccl::allgatherv((int8_t *)name.data(), name.size(), (int8_t *)allNames.data(), aReceiveCount, comm).wait();
    int localRank = 0;
    for (int i = 0; i < rank; i++)
    {
        auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
        std::string nbrName(nameBegin, nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
        if (nbrName == str) localRank++;
    }
    return localRank;

    return 0;
}

bool setGPUContext(ccl::communicator &comm, int gpu_idx[], int n_gpu) {
    int rank = comm.rank();
    int size = comm.size();
    
    /* Create GPU device from local rank and set execution context */
    auto local_rank = getLocalRank(comm, size, rank);
    auto gpus       = get_gpus();
    auto rank_gpu   = gpus[gpu_idx[local_rank % n_gpu]];

    cl::sycl::queue queue(rank_gpu);
    daal::services::SyclExecutionContext ctx(queue);
    daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
}
