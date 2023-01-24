#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include <unordered_set>
#include <vector>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
namespace CudaMallocManaged {

// CUDA device allocator that uses CudaMallocManaged to implement
// the same interface as CUDACachingAllocator.cpp.

// Implementation details, not declared in CUDACachingAllocator.h
namespace {

// General helpers

struct UsageStream {
  cudaStream_t stream;
  int device;
  UsageStream() {}
  UsageStream(cudaStream_t s, int d) : stream(s), device(d) {}
  UsageStream(const UsageStream& us) : stream(us.stream), device(us.device) {}
  UsageStream(const UsageStream&& us) : stream(us.stream), device(us.device) {}
  UsageStream& operator=(UsageStream other) {
    stream = other.stream;
    device = other.device;
    return *this;
  }
};

bool operator==(const UsageStream& lhs, const UsageStream& rhs) {
  return (lhs.stream == rhs.stream) && (lhs.device == rhs.device);
}

struct UsageStreamHash {
  size_t operator()(const UsageStream& us) const noexcept {
    return std::hash<void*>{}(us.stream) + size_t(us.device);
  }
};

struct PtrUsage {

  uint64_t size;
  PtrUsage(uint64_t s) : size(s) {}
};

int device_count = 0;
// these don't need to be c10::once_flags as in CUDAGeneratorImpl.cpp
// because they'll only be flipped by functions that have locked the mutex.
bool devs_initialized_flag;

// Possible micro-optimization:
// Some accesses to ptr_info are read-only.
// We could let those be concurrent with a shared_mutex and
// have concurrent calls take a shared_lock.
// Keeping it simple with an ordinary mutex for now.
std::mutex general_mutex;

using PtrInfo = ska::flat_hash_map<void*, PtrUsage>;
PtrInfo ptr_info;


// These two help setMemoryFraction limit the amount of memory
// used by PyTorch in particular (as opposed to other libraries
// in the same process that might be sharing the same cudaMemPool_t).
size_t pytorch_used_bytes;
size_t pytorch_memory_limits;

bool capture_underway = false;

// Implementation functions

// Assumes the caller holds general_mutex
inline void lazy_init_device(int device) {
  if (!devs_initialized_flag) {
    pytorch_used_bytes= 0;
    pytorch_memory_limits = UINT64_MAX;

    devs_initialized_flag = true;
  }
}


void free(void* ptr) {
  std::lock_guard<std::mutex> lk(general_mutex);

  auto err = cudaGetLastError();
  C10_CUDA_CHECK(err);
  auto it = ptr_info.find(ptr);

  TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");
  
  C10_CUDA_CHECK(cudaFree(it->first));

  pytorch_used_bytes -= it->second.size;

  ptr_info.erase(it);
}

// Symmetric with NativeCachingAllocator::malloc for now,
// although I don't think we absolutely need the symmetry.
void mallocManaged(void** devPtr, int device, size_t size, cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(
      0 <= device && device < device_count,
      "Invalid device index ",
      device,
      ": did you call init?");

  CUDAGuard g(device);

  std::lock_guard<std::mutex> lk(general_mutex);

  lazy_init_device(device);

  // Defensively checks for preexisting CUDA error state.
  auto err = cudaGetLastError();
  C10_CUDA_CHECK(err);

  if (pytorch_used_bytes + size > pytorch_memory_limits) {
    err = cudaErrorMemoryAllocation;
  } else {
    err = cudaMallocManaged(devPtr, size);
  }

  if (err == cudaErrorMemoryAllocation) {
    // Clears CUDA's internal error state so the user, if desired, can catch the
    // OOM exception, free some stuff on the script side, and retry the
    // allocation. This aligns with the behavior of alloc_block in
    // CUDACachingAllocator.cpp.
    cudaGetLastError();
    size_t device_free;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        false,
        "Allocation on device ",
        device,
        " would exceed allowed memory. (out of memory)",
        "\nCurrently allocated     : ",
        format_size(pytorch_used_bytes),
        "\nRequested               : ",
        format_size(size),
        "\nDevice limit            : ",
        format_size(device_total),
        "\nFree (according to CUDA): ",
        format_size(device_free),
        "\nPyTorch limit (set by user-supplied memory fraction)"
        "\n                        : ",
        format_size(pytorch_memory_limits));
  } else {
    C10_CUDA_CHECK(err);
  }
auto inserted = ptr_info.emplace(*devPtr,  PtrUsage(size));
  TORCH_INTERNAL_ASSERT(
      inserted.second,
      "address returned by cudaMallocManaged already exists "
      "in ptr_info");
  pytorch_used_bytes += size;
}

} // anonymous namespace

void local_raw_delete(void* ptr);

// Same pattern as CUDACachingAllocator.cpp.
struct CudaMallocManagedAllocator : public CUDAAllocator {
  DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      mallocManaged(&r, device, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &local_raw_delete, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &local_raw_delete;
  }

  // This function should not issue any context-creating calls,
  // just set up for later calls to init per-device pools based
  // on the current device each later call sees.
  void init(int dev_count) override {
    static bool called = [](int dev_count) {
      ;
      // Are there external guarantees init will be called before
      // any of the allocator's other functions?
      // std::lock_guard<std::mutex> lk(general_mutex);
      device_count = dev_count;
      return true;
    }(dev_count);
    (void)called;
  }

  bool initialized() override {
    return devs_initialized_flag;
  }

  static inline void assertValidDevice(int device) {
    TORCH_CHECK(
        0 <= device && device < device_count, "Invalid device argument.");
  }

  void setMemoryFraction(double fraction, int device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");

    std::lock_guard<std::mutex> lk(general_mutex);
    assertValidDevice(device);
    CUDAGuard g(device);
    // Should setMemoryFraction be allowed to trigger a full device context and
    // pool-creating lazy_init_device, or should we simply assert this device is
    // already initialized, ie
    // TORCH_CHECK(devs_initialized_flags[device], ...)?
    lazy_init_device(device);

    size_t device_free;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    pytorch_memory_limits =
        static_cast<uint64_t>(fraction * device_total);

    // Alternative: Instead of a manual hard limit, we could use
    // cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold,
    // &threshold); This is a soft hint: The driver allows the pool's reserved
    // memory to spike above threshold in regions of high cudaMallocAsync
    // demand, but opportunistically trims reserved memory back to threshold
    // when the memory in use is < threshold. I don't like this because it
    // introduces performance nondeterminism.
  }

  void emptyCache(void) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    for (int dev = 0; dev < device_count; dev++) {
      if (devs_initialized_flag) {
        CUDAGuard g(dev);

        cudaMemPool_t mempool;
        cudaDeviceGetDefaultMemPool(&mempool, dev);
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(mempool, 0);
      }
    }
  }

  void cacheInfo(int device, size_t* maxWorkspaceGuess) override {
    // The only consumer of cacheInfo is getMaxWorkspaceSize in Conv_v7.cpp.
    // Afaict, the role of cacheInfo is to give getMaxWorkspaceSize a reasonable
    // maximum workspace size to use for an upcoming cudnnFind call.
    //
    // The native allocator's cacheInfo chooses to return the size of its
    // largest unused block (which is the largest allocation the native
    // allocator can service immediately and asynchronously without a
    // cudaMalloc.
    //
    // Here, we use a different heuristic: figure out the max usable workspace
    // size with a bit of educated trial and error. It's ok to be
    // perf-inefficient because cacheInfo is a prelude to cudnnFind.
    //
    // The algo cache then stores the best-performing algo with workspace <=
    // maxWorkspaceGuess. Later calls with the same param set hit in cache and
    // try to allocate the same workspace. If, in one of those future calls,
    // workspace allocation fails (ie because less ambient memory is available),
    // the bindings rerun cudnnFind, including calling cacheInfo again
    // beforehand to estimate a new (smaller) largest-available workspace. Over
    // a few such calls, the cache should settle to the algo with a workspace
    // size that's small enough to succeed every time (for that param set).
    //
    // So the strategy here is to return a rough, largeish guess and let the
    // bindings retry to trim as needed over time.

    std::lock_guard<std::mutex> lk(general_mutex);
    assertValidDevice(device);
    CUDAGuard g(device);
    lazy_init_device(device);

    size_t free_upper_bound;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_upper_bound, &device_total));
    TORCH_INTERNAL_ASSERT(
        free_upper_bound + pytorch_used_bytes <= device_total);
    size_t guess = std::min(
        free_upper_bound,
        pytorch_memory_limits - pytorch_used_bytes);
    auto stream = c10::cuda::getCurrentCUDAStream();
    void* dummy;

    // Defensively checks for preexisting CUDA error state.
    auto err = cudaGetLastError();
    C10_CUDA_CHECK(err);

    while (true) {
      // Duplicates some logic from mallocAsync to work with the error state
      // directly instead of repeatedly catching an exception thrown by
      // mallocAsync.
      if (pytorch_used_bytes + guess > pytorch_memory_limits) {
        err = cudaErrorMemoryAllocation;
      } else {
        err = cudaMallocManaged(&dummy, guess);
      }

      if (err == cudaSuccess) {
        cudaFree(dummy);
        *maxWorkspaceGuess = guess;
        return;
      } else if (err == cudaErrorMemoryAllocation) {
        cudaGetLastError(); // clear CUDA error
        guess >>= 1; // quick and dirty: try half the size next iteration
      } else {
        C10_CUDA_CHECK(err);
      }
    }
  }

  void* getBaseAllocation(void* ptr, size_t* size) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    auto it = ptr_info.find(ptr);
    TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

    if (size) {
      *size = it->second.size;
    }

    return ptr;
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support recordStream. "
        "If you need it, please file an issue describing your use case.");
  }

  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support getIpcDevPtr. "
        "If you need it, please file an issue describing your use case.");
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support attachOutOfMemoryObserver. "
        "If you need it, please file an issue describing your use case.");
  }

  // Collects stats for device.
  // If device hasn't been used yet, returns 0s without creating a context.
  DeviceStats getDeviceStats(int device) override {
    assertValidDevice(device);

    // Memory currently reserved by the mempool
    uint64_t reserved_mem_current = 0;
    // High-water mark of memory reserved by the mempool since last reset
    uint64_t reserved_mem_peak = 0;
    // Memory currently in use by the mempool
    uint64_t used_mem_current = 0;
    // High-water mark of memory
    uint64_t used_mem_peak = 0;

    std::lock_guard<std::mutex> lk(general_mutex);

    if (devs_initialized_flag) {
      CUDAGuard g(device);

      cudaMemPool_t mempool;
      C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
      C10_CUDA_CHECK(cudaMemPoolGetAttribute(
          mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem_current));

      C10_CUDA_CHECK(cudaMemPoolGetAttribute(
          mempool, cudaMemPoolAttrReservedMemHigh, &reserved_mem_peak));

      C10_CUDA_CHECK(cudaMemPoolGetAttribute(
          mempool, cudaMemPoolAttrUsedMemCurrent, &used_mem_current));

      C10_CUDA_CHECK(cudaMemPoolGetAttribute(
          mempool, cudaMemPoolAttrUsedMemHigh, &used_mem_peak));
    }

    // Many stat types are specific to the native allocator. We leave these
    // untouched. Their "struct Stat"s will contain zeroed values.
    DeviceStats stats;

    // In the native allocator:
    // allocated_bytes is the total bytes of blocks that have been malloc()ed
    // and not yet free()d.
    // active_bytes is the total bytes of blocks that have been malloc()ed but
    // not yet released back into a free pool. In other words, it includes all
    // allocated_bytes, as well as the bytes of "limbo state" blocks had have
    // already been free()ed but not yet free_block()ed back into a pool due to
    // outstanding stream_uses.
    //
    // Here, in the CudaMallocManaged allocator:
    // We simply ask the driver's opinion about active memory.
    // We don't bother distinguishing between allocated_bytes and active_bytes.
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        used_mem_current;
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        used_mem_peak;
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        used_mem_current;
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        used_mem_peak;
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        reserved_mem_current;
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        reserved_mem_peak;

    return stats;
  }

  void resetAccumulatedStats(int device) override {
    assertValidDevice(device);
    TORCH_WARN_ONCE(
        "For backend:cudaMallocManaged, resetAccumulatedStats has no effect.");
  }

  void resetPeakStats(int device) override {
    assertValidDevice(device);

    CUDAGuard g(device);
    cudaMemPool_t mempool;
    C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
    // Using zero as the reset value is the method recommended by Cuda driver
    // team. Vivek Kini says:
    //   "Resetting to zero (which is the only valid value when setting
    //    ReservedMemHigh) resets it to ReservedMemCurrent inside the driver
    //   (same goes for UsedMemHigh/UsedMemCurrent)"
    uint64_t zero = 0;
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool, cudaMemPoolAttrReservedMemHigh, &zero));
    C10_CUDA_CHECK(
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrUsedMemHigh, &zero));
  }

  SnapshotInfo snapshot() override {
    TORCH_CHECK(
        false,
        "Calling snapshot with backend:cudaMallocManaged is not supported. ");
    // Alternative: TORCH_WARN
    return {};
  }

  // CUDAGraph interactions
  void notifyCaptureBegin(
      int device,
      CaptureId_t graph_id,
      MempoolId_t mempool_id) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support notifyCaptureBegin. "
        "If you need it, please file an issue describing your use case.");
  }

  void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support notifyCaptureAboutToEnd. "
        "If you need it, please file an issue describing your use case.");
  }

  void notifyCaptureEnded(int device, CaptureId_t graph_id) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support notifyCaptureEnded. "
        "If you need it, please file an issue describing your use case.");
  }

  void notifyCaptureDestroy(int device, MempoolId_t mempool_id) override {
    TORCH_CHECK(
        false,
        "cudaMallocManaged does not yet support notifyCaptureDestroy. "
        "If you need it, please file an issue describing your use case.");
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    mallocManaged(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    mallocManaged(&r, device, nbytes, stream);
    return r;
  }
  void raw_delete(void* ptr) override {
    free(ptr);
  }
  bool needsPoolSpecificPeerAccess() override {
    return true;
  }
  std::string name() override {
    return "CudaMallocManaged";
  }
};

CudaMallocManagedAllocator device_allocator;

void local_raw_delete(void* ptr) {
  free(ptr);
}
CUDAAllocator* allocator() {
  return &device_allocator;
}


} // namespace CudaMallocManaged
} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
