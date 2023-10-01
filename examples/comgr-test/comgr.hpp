#pragma once

#include <amd_comgr/amd_comgr.h>

#include <exception>
#include <string_view>
#include <span>
#include <initializer_list>

namespace amd::comgr
{
    enum class status
    {
        success = AMD_COMGR_STATUS_SUCCESS,
        error = AMD_COMGR_STATUS_ERROR,
        error_invalid_argument = AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
        error_out_of_resources = AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
    };

    enum class language
    {
        none = AMD_COMGR_LANGUAGE_NONE,
        opencl_1_2 = AMD_COMGR_LANGUAGE_OPENCL_1_2,
        opencl_2_0 = AMD_COMGR_LANGUAGE_OPENCL_2_0,
        hc = AMD_COMGR_LANGUAGE_HC,
        hip = AMD_COMGR_LANGUAGE_HIP
    };

    /*! \brief Error class
     *
     *  This may be thrown by COMGR wrapper functions when
     *  the underlying library report an error status
     */
    class error : public std::exception {
    private:
        status status_;
        const char* errStr_;

    public:
        /*! \brief Create a new SDK error exception for a given error code
         *  and corresponding message.
         *
         *  \param s status code.
         *
         *  \param errStr a descriptive string describing the error which shall
         *                outlive the catching of the exception obj. If set, it
         *                will be returned by what().
         */
        error(status s, const char* errStr = NULL): status_(s), errStr_(errStr)
        {}

        ~error() throw() {}

        /*! \brief Get error string associated with exception
         *
         * \return A memory pointer to the error message string.
         */
        virtual const char* what() const throw()
        {
            if (errStr_ == NULL)
            {
                return "empty";
            }
            else
            {
                return errStr_;
            }
        }

        /*! \brief Get status code associated with exception
         *
         *  \return The status code.
         */
        status status(void) const { return status_; }
    };

    class data
    {
    public:
        enum class kind
        {
            // No data is available
            undef = AMD_COMGR_DATA_KIND_UNDEF,
            // The data is a textual main source.
            source = AMD_COMGR_DATA_KIND_SOURCE,
            // The data is a textual source that is included in the main source or other include source.
            include = AMD_COMGR_DATA_KIND_INCLUDE,
            // The data is a precompiled-header source that is included in the main source or other include source.
            precompiled_header = AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER,
            // The data is a diagnostic output.
            diagnostic = AMD_COMGR_DATA_KIND_DIAGNOSTIC,
            // The data is a textual log output.
            log = AMD_COMGR_DATA_KIND_LOG,
            // The data is compiler LLVM IR bit code for a specific isa.
            bc = AMD_COMGR_DATA_KIND_BC,
            // The data is a relocatable machine code object for a specific isa.
            relocatable = AMD_COMGR_DATA_KIND_RELOCATABLE,
            // The data is an executable machine code object for a specific isa. An executable is the kind of code object that can be loaded and executed.
            executable = AMD_COMGR_DATA_KIND_EXECUTABLE,
            // The data is a block of bytes.
            bytes = AMD_COMGR_DATA_KIND_BYTES,
            // The data is a fat binary (clang-offload-bundler output).
            fatbin = AMD_COMGR_DATA_KIND_FATBIN//,
            // The data is an archive.
//            ar = AMD_COMGR_DATA_KIND_AR,
            // The data is a bundled bitcode.
//            bc_bundle = AMD_COMGR_DATA_KIND_BC_BUNDLE,
            // The data is a bundled archive.
//            ar_bundle = AMD_COMGR_DATA_KIND_AR_BUNDLE
        };

        data() = default;
        data(const data&) = delete;
        data(data&& in) { std::swap(data_, in.data_); }
        ~data()
        {
            auto s = amd_comgr_release_data(data_);
            err_handler(static_cast<status>(s), "amd_comgr_release_data");
        }

        data(kind k)
        {
            auto s = amd_comgr_create_data(static_cast<amd_comgr_data_kind_t>(k), &data_);
            err_handler(static_cast<status>(s), "amd_comgr_create_data");
        }

        data(kind k, std::string_view name, std::span<std::byte> data__) : data(k)
        {
            auto s = amd_comgr_set_data_name(data_, name.data());
            err_handler(static_cast<status>(s), "amd_comgr_set_data_name");

            s = amd_comgr_set_data(data_, data__.size_bytes(), reinterpret_cast<const char*>(data__.data()));
            err_handler(static_cast<status>(s), "amd_comgr_create_data");
        }

        amd_comgr_data_t get() const { return data_; }
    private:
        amd_comgr_data_t data_;

        void err_handler(status s, const char* src)
        {
            if (s != status::success)
                throw error{s, src};
        }
    };

    class dataset
    {
    public:
        dataset() = default;
        dataset(const dataset&) = delete;
        dataset(dataset&& in) { std::swap(dataset_, in.dataset_); }
        ~dataset()
        {
            auto s = amd_comgr_destroy_data_set(dataset_);
            err_handler(static_cast<status>(s), "amd_comgr_release_data_set");
        }

        dataset(std::initializer_list<data> in)
        {
            for (const data& dat : in)
            {
                auto s = amd_comgr_data_set_add(dataset_, dat.get());
                err_handler(static_cast<status>(s), "amd_comgr_data_set_add");
            }
        }
    private:
        amd_comgr_data_set_t dataset_;

        void err_handler(status s, const char* src)
        {
            if (s != status::success)
                throw error{s, src};
        }
    };

    class action
    {
        void operator()()
        {

        }
    };
}
