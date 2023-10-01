#include <CL/opencl.hpp>

#include <vector>     // std::vector
#include <exception>  // std::runtime_error, std::exception
#include <iostream>   // std::cout
#include <fstream>    // std::ifstream
#include <random>     // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>  // std::transform
#include <cstdlib>    // EXIT_FAILURE
#include <execution>  // std::execution::par_unseq, std::execution::seq
#include <stdexcept>
#include <filesystem> // std::filesystem::canonical
#include <optional>
#include <sstream>
#include <string_view>

#include <CL/Utils/Context.hpp> // cl::util::get_context
#include <CL/Utils/Event.hpp>   // cl::util::get_duration

#include <llvm/IR/InstIterator.h>

#ifdef _MSC_VER
#pragma warning( push, 0 )
#endif
#define AMD_COMGR_STATIC
#include "comgr.hpp"

#include <sstream>

void checkLogs(const char *id, amd_comgr_data_set_t dataSet)
{
  amd_comgr_status_t status;

  size_t count;
  status = amd_comgr_action_data_count(dataSet, AMD_COMGR_DATA_KIND_LOG, &count);
  if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_count(AMD_COMGR_DATA_KIND_LOG)"};

  for (size_t i = 0; i < count; i++) {
    amd_comgr_data_t data;
    status = amd_comgr_action_data_get_data(dataSet, AMD_COMGR_DATA_KIND_LOG, i,
                                            &data);
    if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_get_data(AMD_COMGR_DATA_KIND_LOG)"};

    size_t size;
    status = amd_comgr_get_data(data, &size, NULL);
    if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data"};

    std::string data_log(size, '\0');
    status = amd_comgr_get_data(data, &size, data_log.data());
    if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data"};

    std::cerr << "id: " << id << " has log " << i << std::endl;
    std::cerr << data_log << std::endl;

    status = amd_comgr_release_data(data);
    if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_release_data"};
  }
}

#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
//#include <LLVMSPIRVLib.h>

// private header
//#include <SPIRVModule.h>

// llc-based codegen
#include <llvm/Support/InitLLVM.h>              // llvm::InitLLVM()
#include <llvm/Support/TargetSelect.h>          // llvm::InitializeAllTargets(), InitializeAllTargetMCs()
#include <llvm/PassRegistry.h>                  // llvm::PassRegistry::getPassRegistry()
#include <llvm/InitializePasses.h>              // llvm::InitializeCore()
#include <llvm/Analysis/TargetLibraryInfo.h>    // llvm::
#include <llvm/ADT/Triple.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/CodeGen/CommandFlags.h>

// zig-based codegen
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/InitializePasses.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Object/Archive.h>
#include <llvm/Object/ArchiveWriter.h>
#include <llvm/Object/COFF.h>
#include <llvm/Object/COFFImportFile.h>
#include <llvm/Object/COFFModuleDefinition.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetParser.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/CodeGenCWrappers.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Instrumentation/ThreadSanitizer.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/AddDiscriminators.h>
#include <llvm/Transforms/Utils/CanonicalizeAliases.h>
#include <llvm/Transforms/Utils/NameAnonGlobals.h>

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#include <future>
#include <iterator>

namespace spirv2bin
{
struct device_comparator
{
    bool operator()(const cl::Device& lhs, const cl::Device& rhs) const { return lhs() < rhs(); }
};
struct layer
{
public:
    layer()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (auto& platform : platforms)
        {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (auto& device : devices)
            {
                device_ISA_names.insert({
                    device,
                    std::async(std::launch::async, [&, this](){ return get_ISA(device); })
                });
            }
        }
    }
private:
    using Binary = cl::Program::Binaries::value_type;

    std::optional<std::string> get_ISA(const Binary& binary) noexcept
    {
        try
        {
            amd_comgr_status_t status;

            amd_comgr_data_t exec_data;
            status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &exec_data);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data"};

            status = amd_comgr_set_data(exec_data, binary.size(), reinterpret_cast<const char*>(binary.data()));
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_set_data"};

            size_t isa_name_size;
            status = amd_comgr_get_data_isa_name(exec_data, &isa_name_size, nullptr);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data_isa_name"};

            std::string isa_name(isa_name_size + 1, '\0');
            status = amd_comgr_get_data_isa_name(exec_data, &isa_name_size, isa_name.data());
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data_isa_name"};

            isa_name.pop_back(); // pop null-terminator
            return isa_name;
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return std::nullopt;
        }
    };

    std::optional<std::string> get_ISA(cl::Device device)
    {
        // TOOD: Check for availability of online compiler
        cl::Context context(device);
        cl::Program program{context, dummy_source};
        program.build({device});

        return get_ISA(program.getInfo<CL_PROGRAM_BINARIES>().at(0));
    }

    std::map<cl::Device, std::future<std::optional<std::string>>, device_comparator> device_ISA_names;
    std::string dummy_source =
        R"(kernel void a(global size_t* x){ x[get_global_id(0)] = get_global_id(0); })";
};
} // namespace spirv2bin

int main(int argc, char *argv[])
{
    try
    {
        // Platform & device selection
        cl::Context context =
            cl::util::get_context(
                argc > 1 ? std::atoi(argv[1]) : 0,
                argc > 2 ? std::atoi(argv[2]) : 0,
                CL_DEVICE_TYPE_ALL);
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl::CommandQueue queue{context, device, cl::QueueProperties::Profiling};

        std::cout << "Selected platform: " << cl::Platform{device.getInfo<CL_DEVICE_PLATFORM>()}.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Load program source
        auto kernel_path = std::filesystem::canonical(argv[0]).parent_path().append("OpenCL-Cpp-SAXPY.cl");
        std::cout << "Compiling dummy kernel: " << kernel_path << std::endl;
        std::ifstream source_file{kernel_path};
        if (!source_file.is_open())
            throw std::runtime_error{std::string{"Cannot open kernel source: "} + kernel_path.generic_string()};

        // Create program and kernel
        cl::Program program{context, std::string{std::istreambuf_iterator<char>{source_file},
                                                 std::istreambuf_iterator<char>{}}};
        program.build({device});

        using Binary = cl::Program::Binaries::value_type;
        auto getIsaName = [](const Binary& binary)
        {
            amd_comgr_status_t status;

            amd_comgr_data_t exec_data;
            status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &exec_data);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data"};

            status = amd_comgr_set_data(exec_data, binary.size(), reinterpret_cast<const char*>(binary.data()));
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_set_data"};

            size_t isa_name_size;
            status = amd_comgr_get_data_isa_name(exec_data, &isa_name_size, nullptr);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data_isa_name"};

            --isa_name_size; // We don't care about the null terminator
            std::string isa_name(isa_name_size, 'X');
            status = amd_comgr_get_data_isa_name(exec_data, &isa_name_size, isa_name.data());
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data_isa_name"};

            return isa_name;
        };

        auto isa_name = getIsaName(program.getInfo<CL_PROGRAM_BINARIES>().at(0));
        std::cout << "ISA name obtained from comgr: " << isa_name << std::endl;
/*
        // For debug
        std::ofstream real_amd_binary{"real_amd_binary.elf", std::ios::binary};
        auto bin = program.getInfo<CL_PROGRAM_BINARIES>().at(0);
        std::copy(bin.begin(), bin.end(), std::ostream_iterator<char>(real_amd_binary));
        real_amd_binary.close();
*/
        auto binary_path = std::filesystem::canonical(argv[0]).parent_path().append("OpenCL-Cpp-SAXPY.spv");
        std::ifstream binary_file{binary_path, std::ios::binary};
        if (!binary_file.is_open())
            throw std::runtime_error{std::string{"Cannot open kernel binary: "} + binary_path.generic_string()};
/*
        // A slightly more explicit, two-step conversion instead of the readSpirv shorthand
        std::string spirv_err;
        auto spirv_module = SPIRV::readSpirvModule(binary_file, spirv_err);
        if (!spirv_module)
            throw std::runtime_error{std::string{"Failed to read SPIRV module: \n"} + spirv_err};

        llvm::LLVMContext llvm_context;
        llvm_context.setOpaquePointers(true); // EmitOpaquePointers
        std::string conversion_err;
        auto llvm_module = llvm::convertSpirvToLLVM(llvm_context, *spirv_module, conversion_err);
        if (!llvm_module)
            throw std::runtime_error{std::string{"Failed to convert SPIRV module: \n"} + conversion_err};
*/

        llvm::LLVMContext llvm_context;
        llvm_context.setOpaquePointers(true); // EmitOpaquePointers
        std::string conversion_err;

        llvm::Module* llvm_module;
        if (!readSpirv(llvm_context, binary_file, llvm_module, conversion_err))
            throw std::runtime_error{std::string{"Failed to convert SPIRV module: \n"} + conversion_err};

        // TODO: Make this a proper pass?
        for (llvm::Function& F : llvm_module->functions()) {
            if (F.getCallingConv() == llvm::CallingConv::SPIR_FUNC) {
                F.setCallingConv(llvm::CallingConv::Fast);
            }

            for (auto I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
                llvm::Instruction* inst = &*I;
                if (auto call = llvm::dyn_cast<llvm::CallInst>(inst)) {
                    if (call->getCallingConv() == llvm::CallingConv::SPIR_FUNC) {
                        call->setCallingConv(llvm::CallingConv::Fast);
                    }
                }
            }
        }

        if (llvm_module->materializeAll())
            throw std::runtime_error{"Failed to materialize SPIRV module"};

        auto parse_isa = [](const std::string_view isa_name)
        {
            auto isa_substr = [&](const size_t start, const size_t end)
            {
               if (end == std::string::npos)
                   return isa_name.substr(start);
               else
                   return isa_name.substr(start, end - start);
            };

            auto arch_vendor_separator_pos = isa_name.find('-', 0);
            auto vendor_os_separator_pos = isa_name.find('-', arch_vendor_separator_pos + 1);
            auto os_environment_separator_pos = isa_name.find('-', vendor_os_separator_pos + 1);
            auto environment_target_separator_pos = isa_name.find('-', os_environment_separator_pos + 1);

            std::stringstream features;

            const auto target_feature_separator_pos = isa_name.find(':', environment_target_separator_pos + 1);
            auto feature_start_pos = target_feature_separator_pos;
            while (feature_start_pos != std::string::npos) {
                auto next_feature_separator_pos = isa_name.find(':', feature_start_pos + 1);
                auto feature = isa_substr(feature_start_pos + 1, next_feature_separator_pos);
                const bool first = feature_start_pos != target_feature_separator_pos;
                feature_start_pos = next_feature_separator_pos;

                if (feature.size() == 0)
                    continue;

                if (first)
                    features << ',';

                if (feature.back() == '+')
                    features << '+' << feature.substr(0, feature.size() - 1);
                else if (feature.back() == '-')
                    features << '-' << feature.substr(0, feature.size() - 1);
            }

            return std::make_tuple(
                llvm::Triple{
                    isa_substr(0, arch_vendor_separator_pos),
                    isa_substr(arch_vendor_separator_pos + 1, vendor_os_separator_pos),
                    isa_substr(vendor_os_separator_pos + 1, os_environment_separator_pos),
                    isa_substr(os_environment_separator_pos + 1, environment_target_separator_pos)
                },
                std::string(isa_substr(environment_target_separator_pos + 1, target_feature_separator_pos)),
                features.str()
            );
        };
        auto [triple, gfx, features] = parse_isa(isa_name);
        std::cout << "Triple as parsed from ISA name: " << triple.str() << std::endl;
        std::cout << "gfx as parsed from ISA name: " << gfx << std::endl;
        std::cout << "LLVM CPU features: " << features << std::endl;

        // zig-based codegen
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetInfos(); // not from zig
        llvm::InitializeAllTargetMCs();
        //llvm::InitializeAllDisassemblers(); // not from zig

        auto CanonCPUName =
        llvm::AMDGPU::getArchNameAMDGCN(llvm::AMDGPU::parseArchAMDGCN(gfx));

        std::cout << "CanonCPUName: " << CanonCPUName.str() << std::endl;

        //llvm_module->setTargetTriple(triple.normalize());
        llvm_module->setTargetTriple(triple.str());

        llvm::SmallVector<llvm::StringRef> arch_list;
        llvm::AMDGPU::fillValidArchListAMDGCN(arch_list);

        std::string lookup_err;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(
            triple.getTriple(),
            lookup_err);
        if (!target)
            throw std::runtime_error{std::string{"Failed to lookup target: \n"} + lookup_err};

        auto target_machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
            triple.getTriple(),
            gfx.c_str(),
            features.c_str(),
            llvm::TargetOptions{}, // Options
            llvm::Reloc::Model::DynamicNoPIC
            //std::nullopt,
            //std::nullopt,
            //llvm::CodeGenOpt::Level::Default
            ));
        if (!target_machine)
            throw std::runtime_error{std::string{"Failed to allocate target machine"}};
        target_machine->setO0WantsFastISel(true);

        // Pipeline configurations
        llvm::PipelineTuningOptions pipeline_opts;
        pipeline_opts.LoopUnrolling = true;
        pipeline_opts.SLPVectorization = true;
        pipeline_opts.LoopVectorization = true;
        pipeline_opts.LoopInterleaving = true;
        pipeline_opts.MergeFunctions = true;

        // Instrumentations
        llvm::PassInstrumentationCallbacks instr_callbacks;
        llvm::StandardInstrumentations std_instrumentations{llvm_context, false};
        std_instrumentations.registerCallbacks(instr_callbacks);

        llvm::PassBuilder pass_builder(target_machine.get(), pipeline_opts,
                                       std::nullopt, &instr_callbacks);

        llvm::LoopAnalysisManager loop_am;
        llvm::FunctionAnalysisManager function_am;
        llvm::CGSCCAnalysisManager cgscc_am;
        llvm::ModuleAnalysisManager module_am;

        // Register the AA manager first so that our version is the one used
        function_am.registerPass([&] {
          return pass_builder.buildDefaultAAPipeline();
        });

        llvm::Triple target_triple(llvm_module->getTargetTriple());
        auto tlii = std::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
        function_am.registerPass([&] { return llvm::TargetLibraryAnalysis(*tlii); });

        // Initialize the AnalysisManagers
        pass_builder.registerModuleAnalyses(module_am);
        pass_builder.registerCGSCCAnalyses(cgscc_am);
        pass_builder.registerFunctionAnalyses(function_am);
        pass_builder.registerLoopAnalyses(loop_am);
        pass_builder.crossRegisterProxies(loop_am, function_am,
                                          cgscc_am, module_am);
        pass_builder.registerPipelineStartEPCallback(
            [](llvm::ModulePassManager &module_pm, llvm::OptimizationLevel) {
                module_pm.addPass(
                    createModuleToFunctionPassAdaptor(llvm::AddDiscriminatorsPass()));
        });

        llvm::ModulePassManager module_pm;

        module_pm = pass_builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

        llvm::legacy::PassManager codegen_pm;
        codegen_pm.add(llvm::createSPIRVToOCL20Legacy());
        codegen_pm.add(llvm::createTargetTransformInfoWrapperPass(target_machine->getTargetIRAnalysis()));
/*
        // For debug
        std::string new_elf;
        llvm::raw_string_ostream elf_output{new_elf};
        target_machine->addPassesToEmitFile(codegen_pm, elf_output, nullptr, llvm::CGFT_ObjectFile);
*/
        // Optimization phase
        module_pm.run(*llvm_module, module_am);

        // Code generation phase
        codegen_pm.run(*llvm_module);

        std::string amd_bitcode;
        llvm::raw_string_ostream module_output{amd_bitcode};
        llvm::WriteBitcodeToFile(*llvm_module, module_output);

        std::ofstream amd_bitcode_file{"amd_bitcode.bc", std::ios::binary};
        amd_bitcode_file << amd_bitcode;
        amd_bitcode_file.close();

        auto getELFbinary = [&](std::span<std::byte> llvm_bitcode, llvm::StringRef comgr_isa_name)//const std::string& comgr_isa_name)
        {
/*
            // Started working on a C++ wrapper to shorten this scope
            using namespace amd::comgr;

            data bitcode_data{data::kind::bc, "saxpy.bc", llvm_bitcode};
*/
            amd_comgr_status_t status;

            amd_comgr_data_t bc_data;
            status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &bc_data);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data"};

            status = amd_comgr_set_data(bc_data, llvm_bitcode.size(), reinterpret_cast<const char*>(llvm_bitcode.data()));
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_set_data"};

            status = amd_comgr_set_data_name(bc_data, "SAXPY_BC");
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_set_data_name"};

            amd_comgr_data_set_t bc_dataset;
            status = amd_comgr_create_data_set(&bc_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data_set"};

            status = amd_comgr_data_set_add(bc_dataset, bc_data);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_data_set_add"};

            amd_comgr_action_info_t action_info;
            status = amd_comgr_create_action_info(&action_info);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_action_info"};

            status = amd_comgr_action_info_set_language(action_info, AMD_COMGR_LANGUAGE_OPENCL_1_2); // AMD_COMGR_LANGUAGE_NONE
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_info_set_language"};

            status = amd_comgr_action_info_set_isa_name(action_info, comgr_isa_name.str().c_str());
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_info_set_isa_name"};

            amd_comgr_data_set_t device_libs_dataset;
            status = amd_comgr_create_data_set(&device_libs_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data_set"};

            std::vector<const char*> device_lib_options = {"finite_only", "unsafe_math", "code_object_v5"};
            status = amd_comgr_action_info_set_option_list(action_info, device_lib_options.data(), device_lib_options.size());
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_info_set_option_list(device_lib_options)"};

            status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES, action_info, bc_dataset, device_libs_dataset);
            checkLogs("bc_dataset AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES", bc_dataset);
            checkLogs("device_libs_dataset AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES", device_libs_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES)"};

            size_t count;
            status = amd_comgr_action_data_count(device_libs_dataset, AMD_COMGR_DATA_KIND_BC, &count);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_count(device_libs_dataset, AMD_COMGR_DATA_KIND_BC)"};

            std::cout << "device_libs_dataset has " << count << " AMD_COMGR_DATA_KIND_BC elements." << std::endl;

            amd_comgr_data_set_t linked_dataset;
            status = amd_comgr_create_data_set(&linked_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data_set"};

            std::vector<const char*> codegen_options = {"-mllvm", "-amdgpu-early-inline-all", "-mcode-object-version=5"};
            status = amd_comgr_action_info_set_option_list(action_info, codegen_options.data(), codegen_options.size());
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_info_set_option_list(linked_dataset)"};

            status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, action_info, device_libs_dataset, linked_dataset);
            checkLogs("device_libs_dataset AMD_COMGR_ACTION_LINK_BC_TO_BC", device_libs_dataset);
            checkLogs("AMD_COMGR_ACTION_LINK_BC_TO_BC", linked_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC)"};

            amd_comgr_data_set_t asm_dataset;
            status = amd_comgr_create_data_set(&asm_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data_set"};

            status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                               action_info, linked_dataset, asm_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY)"};

            status = amd_comgr_action_data_count(asm_dataset, AMD_COMGR_DATA_KIND_SOURCE, &count);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_count"};

            if (count != 1) throw std::runtime_error{"AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY failed."};

            amd_comgr_data_set_t reloc_dataset;
            status = amd_comgr_create_data_set(&reloc_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data_set"};

            status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, asm_dataset, reloc_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE)"};

            status = amd_comgr_action_data_count(reloc_dataset, AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_count"};

            if (count != 1) throw std::runtime_error{"AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE failed."};

            amd_comgr_data_set_t exec_dataset;
            status = amd_comgr_create_data_set(&exec_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_create_data_set"};

            status = amd_comgr_action_info_set_option_list(action_info, NULL, 0);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_info_set_option_list"};

            status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, reloc_dataset, exec_dataset);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_do_action"};

            status = amd_comgr_action_data_count(exec_dataset, AMD_COMGR_DATA_KIND_EXECUTABLE, &count);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_count"};

            if (count != 1) throw std::runtime_error{"AMD_COMGR_DATA_KIND_EXECUTABLE failed."};

            amd_comgr_data_t exec_data;
            status = amd_comgr_action_data_get_data(exec_dataset, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &exec_data);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_action_data_get_data"};

            status = amd_comgr_get_data(exec_data, &count, nullptr);
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data"};

            Binary binary(count);
            status = amd_comgr_get_data(exec_data, &count, reinterpret_cast<char*>(binary.data()));
            if (status != AMD_COMGR_STATUS_SUCCESS) throw std::runtime_error{"amd_comgr_get_data"};

            return binary;
        };

        auto amd_binary = getELFbinary(
            std::span<std::byte>(
                reinterpret_cast<std::byte*>(amd_bitcode.data()),
                reinterpret_cast<std::byte*>(amd_bitcode.data() + amd_bitcode.size())),
            isa_name);

        cl::Program binary_program{
            context,
            { device },
            cl::Program::Binaries{ amd_binary }
        };

        binary_program.build({device});
/*
        // llc-based codegen
        llvm::InitLLVM X(argc, argv);
        llvm_module->setTargetTriple(triple.normalize());

        // Enable debug stream buffering.
        llvm::EnableDebugBuffering = true;

        // Initialize targets first, so that --version shows registered targets.
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();

        // Initialize codegen and IR passes used by llc so that the -print-after,
        // -print-before, and -stop-after options work.
        llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();
        llvm::initializeCore(*Registry);
        llvm::initializeCodeGen(*Registry);
        llvm::initializeLoopStrengthReducePass(*Registry);
        llvm::initializeLowerIntrinsicsPass(*Registry);
        llvm::initializeUnreachableBlockElimLegacyPassPass(*Registry);
        llvm::initializeConstantHoistingLegacyPassPass(*Registry);
        llvm::initializeScalarOpts(*Registry);
        llvm::initializeVectorization(*Registry);
        llvm::initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
        llvm::initializeExpandReductionsPass(*Registry);
        llvm::initializeExpandVectorPredicationPass(*Registry);
        llvm::initializeHardwareLoopsPass(*Registry);
        llvm::initializeTransformUtils(*Registry);
        llvm::initializeReplaceWithVeclibLegacyPass(*Registry);
        llvm::initializeTLSVariableHoistLegacyPassPass(*Registry);

        // Initialize debugging passes.
        initializeScavengerTestPass(*Registry);

        std::string lookup_err;
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), lookup_err);
        if (!target)
            throw std::runtime_error{std::string{"Failed to lookup target: \n"} + lookup_err};

        //llvm::TargetOptions target_options = llvm::codegen::InitTargetOptionsFromCodeGenFlags(triple);
        auto target_machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
            triple.getTriple(),
            isa_name.substr(environment_target_separator_pos + 1, isa_name.size() - environment_target_separator_pos),
            "", // Features
            llvm::TargetOptions{}, // Options
            std::nullopt,
            std::nullopt,
            llvm::CodeGenOpt::Level::Default));
        if (!target_machine)
            throw std::runtime_error{std::string{"Failed to allocate target machine"}};

        llvm::legacy::PassManager pass_manager;

        llvm::TargetLibraryInfoImpl TLII(triple);

        llvm::codegen::setFunctionAttributes(
            isa_name.substr(environment_target_separator_pos + 1, isa_name.size() - environment_target_separator_pos),
            "",
            *llvm_module);
*/

        auto device_saxpy = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(program, "saxpy");
        auto host_saxpy = [a = static_cast<float>(argc)](const float &x, const float &y)
        { return a * x + y; };

        const size_t length = 1024;

        // Init computation
        std::cout << "Generating input for vector op... ";
        std::cout.flush();
        std::vector<float> x(length),
                           y(length);
        std::iota(x.begin(), x.end(), 0.f);
        std::iota(y.begin(), y.end(), 0.f);
        cl_float a = static_cast<cl_float>(argc);
        std::cout << "done." << std::endl;

        cl::Buffer buf_x{queue, x.begin(), x.end(), true},
            buf_y{queue, y.begin(), y.end(), false};

        std::cout << "Executing sequentially on host... ";
        std::cout.flush();
        auto seq_start = std::chrono::high_resolution_clock::now();
        std::transform(std::execution::seq, x.cbegin(), x.cend(), y.cbegin(), y.begin(), host_saxpy);
        auto seq_end = std::chrono::high_resolution_clock::now();
        std::cout << "done." << std::endl;

        std::vector<float> seq_ref = y;
        std::iota(y.begin(), y.end(), 0.f); // Reset output

        std::cout << "Executing in parallel on host... ";
        std::cout.flush();
        auto par_start = std::chrono::high_resolution_clock::now();
        std::transform(std::execution::par_unseq, x.cbegin(), x.cend(), y.cbegin(), y.begin(), host_saxpy);
        auto par_end = std::chrono::high_resolution_clock::now();
        std::cout << "done." << std::endl;

        std::vector<float> &par_ref = y;

        // Validate (sequential vs. parallel)
        if (!std::equal(seq_ref.cbegin(), seq_ref.cend(), par_ref.cbegin()))
            throw std::runtime_error{"Sequential and parallel references mismatch."};

        std::cout << "Executing on device... ";
        std::cout.flush();
        auto dev_start = std::chrono::high_resolution_clock::now();
        cl::Event event = device_saxpy(
            cl::EnqueueArgs{
                queue,
                cl::NDRange(length),
                cl::NDRange(32)
            },
            a, buf_x, buf_y);
        event.wait();
        auto dev_end = std::chrono::high_resolution_clock::now();
        std::cout << "done." << std::endl;

        // (Blocking) fetch of results
        std::vector<float> &dev_res = x;
        cl::copy(queue, buf_y, dev_res.begin(), dev_res.end());

        // Validate
        if (!std::equal(seq_ref.cbegin(), seq_ref.cend(), dev_res.cbegin()))
            throw std::runtime_error{"Host reference and device result mismatch."};
        else
        {
            std::cout << "Serial host execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start).count() << "ms.\n"
                      << "Parallel host execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(par_end - par_start).count() << "ms.\n"
                      << "Device execution seen by host: " << std::chrono::duration_cast<std::chrono::milliseconds>(dev_end - dev_start).count() << "ms.\n"
                      << "Device execution seen by device: " << cl::util::get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::milliseconds>(event).count() << "ms." << std::endl;
        }

    }
    catch (cl::BuildError &error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto &log : error.getBuildLog())
        {
            std::cerr << "\tBuild log for device: " << log.first.getInfo<CL_DEVICE_NAME>() << std::endl
                      << std::endl
                      << log.second << std::endl
                      << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error &error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception &error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
