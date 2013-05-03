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

#include "OrderingExternalGPU.h"

/*! \file OrderingExternalGPU.cc
    \brief Implements OrderingExternalGPU class for computing an ordering external potential field on the GPU
*/

/*! Constructor
    \param sysdef system definition
 */
OrderingExternalGPU::OrderingExternalGPU(boost::shared_ptr<SystemDefinition> sysdef, std::vector<Scalar> order_parameters, 
                                         std::vector<int3> lattice_vectors, Scalar interface_width, unsigned int periodicity, std::string log_suffix)
    : OrderingExternal(sysdef, order_parameters, lattice_vectors, interface_width, periodicity, log_suffix), m_block_size(512)
    {
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void OrderingExternalGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, "OrderingExternalGPU");

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    const BoxDim& box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_order_parameters(this->m_order_parameters, access_location::device, access_mode::read);
    ArrayHandle<int3> d_lattice_vectors(this->m_lattice_vectors, access_location::device, access_mode::read);
    
    gpu_compute_ordering_external_forces(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         d_pos.data,
                         box,
                         m_block_size, d_order_parameters.data, m_lattice_vectors.getNumElements(), d_lattice_vectors.data, m_interface_width, m_periodicity);

    if (this->m_prof) this->m_prof->pop();

    }

//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated OrderingExternalGPU class template.
*/
void export_OrderingExternalGPU()
    {
    boost::python::class_<OrderingExternalGPU, boost::shared_ptr<OrderingExternalGPU>, boost::python::bases<OrderingExternal>, boost::noncopyable >
                  ("OrderingExternalGPU", boost::python::init< boost::shared_ptr<SystemDefinition>, 
                   std::vector<Scalar>, std::vector<int3>, Scalar, unsigned int, std::string >())
                  .def("setBlockSize", &OrderingExternalGPU::setBlockSize)
                  ;
    }
