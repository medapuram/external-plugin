/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

#include "OrderingExternalGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif
#include <cuda.h>

/*! \file OrderingExternalGPU.cuh
    \brief Defines templated GPU kernel code for calculating the external forces.
*/

//! Kernel for calculating external forces
/*! This kernel is called to calculate the external forces on all N particles. Actual evaluation of the potentials and
    forces for each particle is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param params per-type array of parameters for the potential

*/
__global__ void gpu_compute_ordering_external_forces_kernel(float4 *d_force,
                                               float *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const BoxDim box,
                                               const Scalar *order_parameters,
                                               const unsigned int n_wave, 
                                               const int3 *lattice_vectors,
                                               const Scalar *interface_widths)
    {
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // read in the position of our particle.
    // (MEM TRANSFER: 16 bytes)
    Scalar4 posi = d_pos[idx];

    // initialize the force to 0
    Scalar3 force = make_scalar3(0.0, 0.0, 0.0);
    Scalar energy = Scalar(0.0);

    Scalar3 Xi = make_scalar3(posi.x, posi.y, posi.z);
    unsigned int typei = __float_as_int(posi.w);
    Scalar order_parameter = order_parameters[typei];

    Scalar3 L = box.getL();
    
    Scalar cosine = Scalar(0.0);
    Scalar3 deriv = make_scalar3(0.0,0.0,0.0);
    for (unsigned int i = 0; i < n_wave; ++i) {
        Scalar3 q = make_scalar3(Scalar(2.0*M_PI)*lattice_vectors[i].x/L.x, 
                                 Scalar(2.0*M_PI)*lattice_vectors[i].y/L.y, 
                                 Scalar(2.0*M_PI)*lattice_vectors[i].z/L.z);
        Scalar arg, sine, clip_parameter;
        arg = dot(Xi, q);
        clip_parameter = Scalar(1.0)/(interface_widths[i]*dot(q, L));
        sine = clip_parameter*sinf(arg);
        deriv = deriv - sine*q; 
        cosine += clip_parameter*cosf(arg);
    }
    Scalar tanH = tanhf(cosine);
    
    energy = order_parameter*tanH;
    
    Scalar sechSq = (Scalar(1.0) - tanH*tanH);
    Scalar f = order_parameter*sechSq;
    force = f*deriv;

    // now that the force calculation is complete, write out the result)
    d_force[idx].x = force.x;
    d_force[idx].y = force.y;
    d_force[idx].z = force.z;
    d_force[idx].w = energy;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i] = Scalar(0.0);
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param external_potential_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential

    This is just a driver function for gpu_compute_external_forces(), see it for details.
*/
cudaError_t gpu_compute_ordering_external_forces(float4 *d_force,
              float *d_virial,
              const unsigned int virial_pitch,
              const unsigned int N,
              const Scalar4 *d_pos,
              const BoxDim& box,
              const unsigned int block_size,
              const Scalar *d_order_parameters, 
              const unsigned int n_wave,
              const int3 *d_lattice_vectors,
              const Scalar *d_interface_widths)
    {
    // setup the grid to run the kernel
    dim3 grid( N / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // bind the position texture
    gpu_compute_ordering_external_forces_kernel
           <<<grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, box, d_order_parameters, n_wave, d_lattice_vectors, d_interface_widths);

    return cudaSuccess;
    }

