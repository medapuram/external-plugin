/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include "OrderingExternal.h"
#include "OrderingExternalGPU.cuh"

/*! \file OrderingExternalGPU.h
    \brief Declares a class for computing an external potential field on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __ORDERING_EXTERNAL_GPU_H__
#define __ORDERING_EXTERNAL_GPU_H__

//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
*/
class OrderingExternalGPU : public OrderingExternal
    {
    public:
        //! Constructs the compute
        OrderingExternalGPU(boost::shared_ptr<SystemDefinition> sysdef, std::vector<Scalar> order_parameters, 
                            std::vector<int3> lattice_vectors, Scalar interface_width, unsigned int periodicity, std::string log_suffix);

        //! Set the block size to execute on the GPU
        /*! \param block_size Size of the block to run on the device
            Performance of the code may be dependant on the block size run
            on the GPU. \a block_size should be set to be a multiple of 32.
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }

    protected:

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! block size
        unsigned int m_block_size;
    };
    
void export_OrderingExternalGPU();

#endif
