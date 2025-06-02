project_sources = files(
    'util.cu',
)

# Define a static library for common CUDA utilities
common_cuda_lib = static_library(
    'util',
    project_sources,
    # Cuda specific arguments
    # Note: You might need to adjust these based on your CUDA toolkit version and requirements
    cuda_args : ['--gpu-architecture=sm_86'], # Example for Turing architecture
    install : false, # Don't install the library directly
)

# Make the library and its headers available to other parts of the project
common_cuda_dep = declare_dependency(
    link_with : common_cuda_lib,
    include_directories : include_directories('.'), # Export common_utils.cuh
)

# This is the key: Make the 'common_cuda_dep' object available via the
# name 'common_cuda_utils' when other build files use dependency()
meson.override_dependency('common_cuda_utils', common_cuda_dep)

