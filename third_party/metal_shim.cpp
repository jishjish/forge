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

    struct KernelSpecs {
        int maxThreadgroupMemoryLength;
        int maxThreadsPerThreadgroupX;
        int maxThreadsPerThreadgroupY;
        int maxThreadsPerThreadgroupZ;
        int recommendedMaxWorkingSetSize;
    };

    struct BufferAllocationData {
        int index;
        char type[16];
        char name[32];
    };

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
        /*
            Compiles MSL (Metal Shading Language) into a MTLLibrary and 
            extracts a specific MTLFunction from it.
        */
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
        /*
            Returns MTLPipelineState* handle (memory address) for encoder dispatch.
            This is the compiled GPU program ready to dispatch. 
        */
        NS::Error* pipeline_error = nullptr;
        MTL::ComputePipelineState* pipeline = ((MTL::Device*)device)->newComputePipelineState((MTL::Function*)func, &pipeline_error);
        if (!pipeline) { std::cerr << "Failed to generate pipeline";}
        return pipeline;
    }


    void* forge_allocate_input_buffer(void* device, void* data_ptr, int byte_length)
    {
        // Allocate an input buffer for dispatch.
        return ((MTL::Device*)device)->newBuffer(data_ptr, byte_length, MTL::ResourceStorageModeShared);
    }

    void* forge_allocate_output_buffer(void* device, int byte_length)
    {
        /*
            Allocate an empty output buffer for dispatch.
        */
        return ((MTL::Device*)device)->newBuffer(byte_length, MTL::ResourceStorageModeShared);
    }

    void* forge_allocate_constant_buffer(void* device, int data_length)
    {
        MTL::Buffer* buf = ((MTL::Device*)device)->newBuffer(
            sizeof(int), 
            MTL::ResourceStorageModeShared
        );
        memcpy(buf->contents(), &data_length, sizeof(int));
        return buf;
    }

    void* forge_read_output_buf(void* out_buf, int data_length)
    {
        /*
        Read the output buffer produced from the dispatch pipeline.
        */

        MTL::Buffer* buf = (MTL::Buffer*)out_buf;
        float* data = (float*)buf->contents();

        // uncomment to stream results
        // for (int i = 0; i < data_length; i++) {
        //     std::cout << data[i] << "\n" ;
        // }
        return data;
    }

    void* forge_dispatch_pipeline(
        void* device, 
        void* pipeline, 
        KernelSpecs* kernel_specs, 
        int data_length, 
        void* data_ptr,
        BufferAllocationData* buffer_alloc_data,
        int buffer_alloc_len,
        int byte_length,
        bool is_buffer
    )
    {
        MTL::Buffer* in_buf = is_buffer 
            ? (MTL::Buffer*)data_ptr 
            : (MTL::Buffer*)forge_allocate_input_buffer(device, data_ptr, byte_length);
        MTL::CommandQueue* queue = ((MTL::Device*)device)->newCommandQueue();
        MTL::CommandBuffer* cmd_buf = queue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmd_buf->computeCommandEncoder();
        encoder->setComputePipelineState((MTL::ComputePipelineState*)pipeline);

        NS::UInteger simd_width = ((MTL::ComputePipelineState*)pipeline)->threadExecutionWidth();
        NS::UInteger max_threads = ((MTL::ComputePipelineState*)pipeline)->maxTotalThreadsPerThreadgroup();
        NS::UInteger threadgroup_size = MIN((NS::UInteger)data_length, (max_threads / simd_width) * simd_width);
        NS::UInteger grid_size = data_length;

        MTL::Buffer* out_buf = nullptr;
        for (int i = 0; i < buffer_alloc_len; i++){
            int index = buffer_alloc_data[i].index;
            const char* type = buffer_alloc_data[i].type;
            MTL::Buffer* buf;

            if (strcmp(type, "input") == 0){
                buf = in_buf;
            } else if (strcmp(type, "constant") == 0){
                buf = (MTL::Buffer*)forge_allocate_constant_buffer(device, data_length);
            } else if (strcmp(type, "output") == 0){
                buf = (MTL::Buffer*)forge_allocate_output_buffer(device, byte_length);
                out_buf = buf;
            } 
            encoder->setBuffer(buf, 0, index);
        }

        encoder->dispatchThreads(
            MTL::Size(grid_size, 1, 1),
            MTL::Size(threadgroup_size, 1, 1)
        );

        encoder->endEncoding();
        cmd_buf->commit();
        cmd_buf->waitUntilCompleted();

        float* out_debug = (float*)out_buf->contents();
        return out_buf;
    }
}

