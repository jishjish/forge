/*
If adding new functions, re build through 
command `bash build_metal.sh`

MTLDevice object is a thin abstraction for a GPU;
you use it to communicate with the GPU.

Metal divides grids into smaller grids calldd `threadgroups`.
*/

#include <iostream>
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"

/*
future function additions
    - MTLSizeMake(arrayLength, 1, 1) - decide how many threads to create / how to organize.

*/
extern "C" {
    void* forge_get_device()
    {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if(!device) return nullptr;
        return device;
    }

    const char* forge_device_name(void* device)
    {
        if (!device) return "Unknown";
        return((MTL::Device*)device)->name()->utf8String();
    }

    uint64_t forge_max_threadgroup_memory(void* device)
    {
        if (!device) return 0.0;
        return ((MTL::Device*)device)->maxThreadgroupMemoryLength();
    }

    uint64_t forge_max_threads_per_group_x(void* device) 
    {
        if (!device) return 0.0;
        return ((MTL::Device*)device)->maxThreadsPerThreadgroup().width;
    }

    uint64_t forge_max_threads_per_group_y(void* device) 
    {
        if (!device) return 0.0;
        return ((MTL::Device*)device)->maxThreadsPerThreadgroup().height;
    }

    uint64_t forge_max_threads_per_group_z(void* device) 
    {
        if (!device) return 0.0;
        return ((MTL::Device*)device)->maxThreadsPerThreadgroup().depth;
    }

    uint64_t forge_recommended_max_working_size(void* device)
    {
        if (!device) return 0.0;
        return ((MTL::Device*)device)->recommendedMaxWorkingSetSize();
    }

    uint64_t forge_supports_family(void* device, int gpuFamily)
    {
        if (!device) return 0.0;
        return ((MTL::Device*)device)->supportsFamily((MTL::GPUFamily)gpuFamily);
    }

    MTL::Function* forge_compile_source(void* device, const char* source_code, const char* function_name)
    {
        if (!device) { std::cerr << "No device present\n"; return nullptr;}
        NS::String* src = NS::String::string(source_code, NS::StringEncoding::UTF8StringEncoding);
        NS::Error* error = nullptr;
        
        // create new library with the source code
        MTL::Library* lib = ((MTL::Device*)device)->newLibrary(src, nullptr, &error);
        if (!lib) { std::cerr << "Issue creating new metal library";}
        
        // define function name
        NS::String* func_name = NS::String::string(function_name, NS::StringEncoding::UTF8StringEncoding);

        // generate the function
        MTL::Function* func = lib->newFunction(func_name);

        // release the func name
        func_name->release();
        if (!func) { std::cerr << "Failed to find shader function.\n";}

        return func;
    }

    MTL::ComputePipelineState* forge_generate_pipeline(void* device, void* func)
    {
        NS::Error* pipeline_error = nullptr;
        MTL::ComputePipelineState* pipeline = ((MTL::Device*)device)->newComputePipelineState((MTL::Function*)func, &pipeline_error);
        if (!pipeline) { std::cerr << "Failed to generate pipeline";}
        return pipeline;
    }
}

