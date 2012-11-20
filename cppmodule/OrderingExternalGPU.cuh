
cudaError_t gpu_compute_ordering_external_forces(float4 *d_force,
              float *d_virial,
              const unsigned int virial_pitch,
              const unsigned int N,
              const Scalar4 *d_pos,
              const BoxDim& box,
              const unsigned int block_size,
              const Scalar *d_order_parameters, 
              const unsigned int n_wave,
              const int3 *d_lattice_vectors);

