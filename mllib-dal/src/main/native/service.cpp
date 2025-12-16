#include "service.h"
#include "Logger.h"
#include "error_handling.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::services;

#ifdef CPU_GPU_PROFILE
std::mutex g_kmtx;
std::vector<HomogenTablePtr> g_HomogenTablePtrVector;
#endif

bool isFull(NumericTableIface::StorageLayout layout) {
    int layoutInt = (int)layout;
    if (packed_mask & layoutInt) {
        return false;
    }
    return true;
}

bool isUpper(NumericTableIface::StorageLayout layout) {
    if (layout == NumericTableIface::upperPackedSymmetricMatrix ||
        layout == NumericTableIface::upperPackedTriangularMatrix) {
        return true;
    }
    return false;
}

bool isLower(NumericTableIface::StorageLayout layout) {
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix ||
        layout == NumericTableIface::lowerPackedTriangularMatrix) {
        return true;
    }
    return false;
}

template <typename T>
void printArray(T *array, const size_t nPrintedCols, const size_t nPrintedRows,
                const size_t nCols, const std::string &message,
                size_t interval = 10) {
    logger::println(logger::INFO, message);
    for (size_t i = 0; i < nPrintedRows; i++) {
        logger::print(logger::INFO, "");
        for (size_t j = 0; j < nPrintedCols; j++) {
            logger::print(logger::NONE, "%*.3f", interval,
                          array[i * nCols + j]);
        }
        logger::println(logger::NONE, "");
    }
    logger::println(logger::INFO, "");
}

template <typename T>
void printArray(T *array, const size_t nCols, const size_t nRows,
                const std::string &message, size_t interval = 10) {
    printArray(array, nCols, nRows, nCols, message, interval);
}

template <typename T>
void printLowerArray(T *array, const size_t nPrintedRows,
                     const std::string &message, size_t interval = 10) {
    logger::println(logger::INFO, message);
    int ind = 0;
    for (size_t i = 0; i < nPrintedRows; i++) {
        logger::print(logger::INFO, "");
        for (size_t j = 0; j <= i; j++) {
            logger::print(logger::NONE, "%*.3f", interval, array[ind++]);
        }
        logger::println(logger::NONE, "");
    }
    logger::println(logger::INFO, "");
}

template <typename T>
void printUpperArray(T *array, const size_t nPrintedCols,
                     const size_t nPrintedRows, const size_t nCols,
                     const std::string &message, size_t interval = 10) {
    logger::println(logger::INFO, message);
    int ind = 0;
    for (size_t i = 0; i < nPrintedRows; i++) {
        logger::print(logger::INFO, "");
        for (size_t j = 0; j < i; j++) {
            logger::print(logger::NONE, "          ");
        }
        for (size_t j = i; j < nPrintedCols; j++) {
            logger::print(logger::NONE, "%*.3f", interval, array[ind++]);
        }
        for (size_t j = nPrintedCols; j < nCols; j++) {
            ind++;
        }
        logger::println(logger::NONE, "");
    }
    logger::println(logger::INFO, "");
}

void printNumericTable(NumericTable *dataTable, const char *message = "",
                       size_t nPrintedRows = 0, size_t nPrintedCols = 0,
                       size_t interval = 10) {
    size_t nRows = dataTable->getNumberOfRows();
    size_t nCols = dataTable->getNumberOfColumns();
    NumericTableIface::StorageLayout layout = dataTable->getDataLayout();

    if (nPrintedRows != 0) {
        nPrintedRows = std::min(nRows, nPrintedRows);
    } else {
        nPrintedRows = nRows;
    }

    if (nPrintedCols != 0) {
        nPrintedCols = std::min(nCols, nPrintedCols);
    } else {
        nPrintedCols = nCols;
    }

    BlockDescriptor<DAAL_DATA_TYPE> block;
    if (isFull(layout) || layout == NumericTableIface::csrArray) {
        dataTable->getBlockOfRows(0, nRows, readOnly, block);
        printArray<DAAL_DATA_TYPE>(block.getBlockPtr(), nPrintedCols,
                                   nPrintedRows, nCols, message, interval);
        dataTable->releaseBlockOfRows(block);
    } else {
        PackedArrayNumericTableIface *packedTable =
            dynamic_cast<PackedArrayNumericTableIface *>(dataTable);
        if (packedTable) {
            packedTable->getPackedArray(readOnly, block);
            if (isLower(layout)) {
                printLowerArray<DAAL_DATA_TYPE>(
                    block.getBlockPtr(), nPrintedRows, message, interval);
            } else if (isUpper(layout)) {
                printUpperArray<DAAL_DATA_TYPE>(block.getBlockPtr(),
                                                nPrintedCols, nPrintedRows,
                                                nCols, message, interval);
            }
            packedTable->releasePackedArray(block);
        } else {
            logger::println(
                logger::INFO,
                "Dynamic cast to PackedArrayNumericTableIface* failed.");
        }
    }
}

void printNumericTable(NumericTable &dataTable, const char *message = "",
                       size_t nPrintedRows = 0, size_t nPrintedCols = 0,
                       size_t interval = 10) {
    printNumericTable(&dataTable, message, nPrintedRows, nPrintedCols,
                      interval);
}

void printNumericTable(const NumericTablePtr &dataTable, const char *message,
                       size_t nPrintedRows, size_t nPrintedCols,
                       size_t interval) {
    printNumericTable(dataTable.get(), message, nPrintedRows, nPrintedCols,
                      interval);
}

size_t serializeDAALObject(SerializationIface *pData, ByteBuffer &buffer) {
    /* Create a data archive to serialize the numeric table */
    InputDataArchive dataArch;

    /* Serialize the numeric table into the data archive */
    pData->serialize(dataArch);

    /* Get the length of the serialized data in bytes */
    const size_t length = dataArch.getSizeOfArchive();

    /* Store the serialized data in an array */
    buffer.resize(length);
    if (length)
        dataArch.copyArchiveToArray(&buffer[0], length);
    return length;
}

SerializationIfacePtr deserializeDAALObject(daal::byte *buff, size_t length) {
    /* Create a data archive to deserialize the object */
    OutputDataArchive dataArch(buff, length);

    /* Deserialize the numeric table from the data archive */
    return dataArch.getAsSharedPtr();
}

ComputeDevice getComputeDeviceByOrdinal(size_t computeDeviceOrdinal) {
    ComputeDevice device = ComputeDevice::uninitialized;
    switch (computeDeviceOrdinal) {
    case 0:
        device = ComputeDevice::host;
        break;
    case 1:
        device = ComputeDevice::cpu;
        break;
    case 2:
        device = ComputeDevice::gpu;
        break;
    default:
        break;
    }
    return device;
}

#ifdef CPU_GPU_PROFILE

void saveHomogenTablePtrToVector(const HomogenTablePtr &ptr) {
    g_kmtx.lock();
    g_HomogenTablePtrVector.push_back(ptr);
    g_kmtx.unlock();
}

NumericTablePtr homegenToSyclHomogen(NumericTablePtr ntHomogen) {
    int nRows = ntHomogen->getNumberOfRows();
    int nColumns = ntHomogen->getNumberOfColumns();

    // printNumericTable(ntHomogen, "ntHomogen:", 10, 10);

    NumericTablePtr ntSycl = HomogenNumericTable<CpuAlgorithmFPType>::create(
        nColumns, nRows, NumericTable::doAllocate);

    // printNumericTable(ntSycl, "ntSycl:", 10, 10);

    BlockDescriptor<CpuAlgorithmFPType> sourceRows;
    ntHomogen->getBlockOfRows(0, ntHomogen->getNumberOfRows(), readOnly,
                              sourceRows);

    BlockDescriptor<CpuAlgorithmFPType> targetRows;
    ntSycl->getBlockOfRows(0, ntSycl->getNumberOfRows(), writeOnly, targetRows);

    // bracets for calling destructor of hostPtr to release lock
    {
        Status st;
        auto hostPtr =
            targetRows.getBuffer().toHost(data_management::writeOnly, st);

        for (int i = 0; i < nRows * nColumns; i++) {
            hostPtr.get()[i] = sourceRows.getBlockPtr()[i];
        }
    }

    // printNumericTable(ntSycl, "ntSycl Result:", 10, 10);

    return ntSycl;
}

HomogenTablePtr createHomogenTableWithArrayPtr(size_t pNumTabData,
                                               size_t numRows, size_t numClos,
                                               sycl::queue queue) {
    double *htableArray = reinterpret_cast<double *>(pNumTabData);
    auto data = sycl::malloc_shared<double>(numRows * numClos, queue);
    queue.memcpy(data, htableArray, sizeof(double) * numRows * numClos).wait();
    return std::make_shared<homogen_table>(
        queue, data, numRows, numClos,
        detail::make_default_delete<const double>(queue));
}
#endif
