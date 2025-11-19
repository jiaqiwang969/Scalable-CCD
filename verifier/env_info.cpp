// Minimal, portable environment collection for Linux; guarded CUDA info if available.

#include "env_info.hpp"

#include <scalable_ccd/config.hpp>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

#ifdef SCALABLE_CCD_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace {

std::string read_first_line(const char* path)
{
    std::ifstream in(path);
    std::string line;
    if (in.good()) {
        std::getline(in, line);
    }
    return line;
}

std::string get_cpu_model()
{
    // Linux: parse /proc/cpuinfo
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        // model name	: Intel(R) Xeon(R) CPU...
        auto pos = line.find("model name");
        if (pos != std::string::npos) {
            auto colon = line.find(':', pos);
            if (colon != std::string::npos) {
                std::string val = line.substr(colon + 1);
                // trim leading spaces
                size_t start = val.find_first_not_of(" \t");
                if (start != std::string::npos)
                    val = val.substr(start);
                return val;
            }
        }
    }
    return "";
}

std::string get_compiler_string()
{
#if defined(__clang__)
    return std::string("Clang ") + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    return std::string("GCC ") + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "Unknown";
#endif
}

} // namespace

namespace verifier {

nlohmann::json collect_env_info()
{
    nlohmann::json j;

    // OS info
    struct utsname uts {};
    if (uname(&uts) == 0) {
        j["os"]["sysname"] = uts.sysname;
        j["os"]["release"] = uts.release;
        j["os"]["version"] = uts.version;
        j["os"]["machine"] = uts.machine;
        j["os"]["nodename"] = uts.nodename;
    }
    {
        // best-effort distro name
        std::ifstream in("/etc/os-release");
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("PRETTY_NAME=", 0) == 0) {
                auto name = line.substr(std::strlen("PRETTY_NAME="));
                if (!name.empty() && name.front() == '"' && name.back() == '"') {
                    name = name.substr(1, name.size() - 2);
                }
                j["os"]["distro"] = name;
                break;
            }
        }
    }

    // CPU info
    j["cpu"]["model"] = get_cpu_model();
    j["cpu"]["logical_cores"] = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        j["memory"]["total_bytes"] = static_cast<long long>(pages) * static_cast<long long>(page_size);
    }

    // Build/Compile info
    j["build"]["project"] = SCALABLE_CCD_NAME;
    j["build"]["version"] = SCALABLE_CCD_VER;
    j["build"]["type"] = SCALABLE_CCD_BUILD_TYPE; // injected by compile_def
    j["build"]["cxx_compiler"] = get_compiler_string();
#ifdef SCALABLE_CCD_USE_DOUBLE
    j["build"]["precision"] = "double";
#else
    j["build"]["precision"] = "float";
#endif
#ifdef SCALABLE_CCD_WITH_CUDA
    j["build"]["with_cuda"] = true;
#else
    j["build"]["with_cuda"] = false;
#endif
#if defined(SCALABLE_CCD_WITH_METAL)
    j["build"]["with_metal"] = true;
#else
    j["build"]["with_metal"] = false;
#endif

#ifdef SCALABLE_CCD_WITH_CUDA
    // CUDA info
    int driver = 0, runtime = 0, device_count = 0;
    (void)cudaDriverGetVersion(&driver);
    (void)cudaRuntimeGetVersion(&runtime);
    cudaGetDeviceCount(&device_count);
    j["cuda"]["driver_version"] = driver;
    j["cuda"]["runtime_version"] = runtime;
    j["cuda"]["device_count"] = device_count;
    if (device_count > 0) {
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, i);
            nlohmann::json d;
            d["name"] = prop.name;
            d["sm"] = { {"major", prop.major}, {"minor", prop.minor} };
            d["multiProcessorCount"] = prop.multiProcessorCount;
            d["totalGlobalMem"] = static_cast<long long>(prop.totalGlobalMem);
            d["sharedMemPerBlock"] = prop.sharedMemPerBlock;
            d["warpSize"] = prop.warpSize;
            d["maxThreadsPerBlock"] = prop.maxThreadsPerBlock;
            d["clockRate_khz"] = prop.clockRate;
            j["cuda"]["devices"].push_back(d);
        }
    }
#endif

    // Metal info (best-effort; only on Apple + when compiled with Metal)
#if defined(SCALABLE_CCD_WITH_METAL)
    j["metal"]["available"] = true;
    j["metal"]["device_name"] = "metal-cpp";
#endif

    return j;
}

} // namespace verifier
