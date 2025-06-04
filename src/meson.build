common_cuda_dep = dependency('common_cuda_utils')

# Define test_vector_add executable
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

# Define test_vector_sum executable
# executable(
#     'test_vector_sum',
#     'test_vector_sum.cu',
#     dependencies : common_cuda_dep, # Link with common_cuda_utils
#     # Cuda specific arguments
#     # Note: You might need to adjust these based on your CUDA toolkit version and requirements
#     cuda_args : ['--gpu-architecture=sm_86'],
#     link_args : ['-lcudart', '-lcuda'],
#     install : true,
# )
