common_cuda_dep = dependency('common_cuda_utils')

executable(
    'test_mat_mul',
    'test_mat_mul.cu',
    dependencies : common_cuda_dep, # Link with common_cuda_utils
    # Cuda specific arguments
    # Note: You might need to adjust these based on your CUDA toolkit version and requirements
    cuda_args : ['--gpu-architecture=sm_86'],
    link_args : ['-lcudart', '-lcuda'], # Link against CUDA runtime and driver libraries
    install : true,
)

executable(
    'bench_mat_mul',
    'bench_mat_mul.cu',
    dependencies : common_cuda_dep, # Link with common_cuda_utils
    # Cuda specific arguments
    # Note: You might need to adjust these based on your CUDA toolkit version and requirements
    cuda_args : ['--gpu-architecture=sm_86'],
    link_args : ['-lcudart', '-lcuda'],
    install : true,
)
