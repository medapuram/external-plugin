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

#include "OrderingExternal.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/*! \file OrderingExternal.cc
    \brief Implements the OrderingExternal class
*/

/*! Constructor
    \param sysdef system definition
 */
OrderingExternal::OrderingExternal(boost::shared_ptr<SystemDefinition> sysdef, std::vector<Scalar> order_parameters, 
                                   std::vector<int3> lattice_vectors, Scalar interface_width, unsigned int periodicity, std::string log_suffix)
    : ForceCompute(sysdef), m_interface_width(interface_width), m_periodicity(periodicity)
    {
    m_log_name = std::string("external_ordering") + log_suffix;

    if(order_parameters.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Number of order parameters is not the same as number of atom types" << std::endl;
        throw std::runtime_error("Error constructing OrderingExternal"); 
        }
    
    GPUArray<Scalar> gpu_order_parameters(m_pdata->getNTypes(), exec_conf);
    m_order_parameters.swap(gpu_order_parameters);
    ArrayHandle<Scalar> h_order_parameters(m_order_parameters, access_location::host, access_mode::overwrite);
    for(unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
        {
        h_order_parameters.data[i] = order_parameters[i];
        }

    GPUArray<int3> gpu_lattice_vectors(lattice_vectors.size(), exec_conf);
    m_lattice_vectors.swap(gpu_lattice_vectors);
    ArrayHandle<int3> h_lattice_vectors(m_lattice_vectors, access_location::host, access_mode::overwrite);
    for(unsigned int i = 0; i < m_lattice_vectors.getNumElements(); ++i)
        {
        h_lattice_vectors.data[i] = lattice_vectors[i];
        }
    }

/*! PotentialExternal provides
    - \c external_"name"_energy
*/
std::vector< std::string > OrderingExternal::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
Scalar OrderingExternal::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "Ordering external: " << quantity << " is not a valid log quantity" << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }


/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void OrderingExternal::computeForces(unsigned int timestep)
    {

    if (m_prof) m_prof->push("OrderingExternal");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar> h_order_parameters(m_order_parameters,access_location::host, access_mode::read);
    ArrayHandle<int3> h_lattice_vectors(m_lattice_vectors,access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();

    unsigned int nparticles = m_pdata->getN();

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

   // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

    unsigned int n_wave = m_lattice_vectors.getNumElements();

    // for each of the particles
    for (unsigned int idx = 0; idx < nparticles; idx++)
        {
        Scalar3 F;
        Scalar energy;

        // get the current particle properties
        Scalar3 X = make_scalar3((h_pos.data[idx].x + (L.x/Scalar(2.0)))/(L.x), 
                                 (h_pos.data[idx].y + (L.y/Scalar(2.0)))/(L.y), 
                                 (h_pos.data[idx].z + (L.z/Scalar(2.0)))/(L.z));
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        Scalar order_parameter = h_order_parameters.data[type];

        Scalar cosine = Scalar(0.0);
        Scalar3 deriv = make_scalar3(0.0,0.0,0.0);
        Scalar clip_parameter = Scalar(1.0)/(Scalar(2.0*M_PI)*m_interface_width*m_periodicity);
        for (unsigned int i = 0; i < n_wave; ++i) 
            {
            Scalar3 q = make_scalar3(2.0*M_PI*h_lattice_vectors.data[i].x/L.x,
                                     2.0*M_PI*h_lattice_vectors.data[i].y/L.y,
                                     2.0*M_PI*h_lattice_vectors.data[i].z/L.z);
            Scalar3 qr = make_scalar3(2.0*M_PI*h_lattice_vectors.data[i].x,
                                      2.0*M_PI*h_lattice_vectors.data[i].y,
                                      2.0*M_PI*h_lattice_vectors.data[i].z);

            Scalar arg, q_length, clip_parameter, sine;
            arg = dot(X, qr);
            q_length = dot(q, L);
            if (h_lattice_vectors.data[i].x != 0 || h_lattice_vectors.data[i].y != 0 || h_lattice_vectors.data[i].z != 0) {
                clip_parameter = Scalar(1.0)/(m_interface_width*q_length);
            } else {
                clip_parameter = Scalar(0.0);
            }
            cosine += clip_parameter*cosf(arg);
            sine = -Scalar(1.0)*clip_parameter*sinf(arg);
            deriv = deriv + sine*q;
            }
        Scalar tanH = tanhf(cosine);
        energy = order_parameter*tanH;

        Scalar sechSq = (1.0 - tanH*tanH);
        Scalar f = order_parameter*sechSq;
        F = f*deriv;

        // apply the constraint force
        h_force.data[idx].x = F.x;
        h_force.data[idx].y = F.y;
        h_force.data[idx].z = F.z;
        h_force.data[idx].w = energy;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*m_virial_pitch+idx]  = 0.0;
        }

    if (m_prof)
        m_prof->pop();
    }

//! Set the parameters for this potential
/*! \param type type for which to set parameters
    \param params value of parameters
*/
void OrderingExternal::setParams(unsigned int type, Scalar order_parameter)
    {
    if (type >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "OrderingExternal: Trying to set external potential params for a non existant type! "
                                        << type << std::endl;
        throw std::runtime_error("Error setting parameters in OrderingExternal");
        }

    ArrayHandle<Scalar> h_order_parameters(m_order_parameters, access_location::host, access_mode::readwrite);
    h_order_parameters.data[type] = order_parameter;
    }

//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated OrderingExternal class template.
*/
void export_OrderingExternal()
    {
    boost::python::class_<OrderingExternal, boost::shared_ptr<OrderingExternal>, boost::python::bases<ForceCompute>, boost::noncopyable >
                  ("OrderingExternal", boost::python::init< boost::shared_ptr<SystemDefinition>, std::vector<Scalar>, 
                   std::vector<int3>, Scalar, unsigned int, std::string >())
                  ;

        class_<std::vector<int3> >("std_vector_int3")
        .def(vector_indexing_suite< std::vector<int3> > ())
        ;
    }


