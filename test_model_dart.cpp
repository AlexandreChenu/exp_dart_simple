//| This file is a part of the sferes2 framework.
//| Copyright 2016, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#include <iostream>

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>


#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/grid.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/tournament.hpp>
#include <sferes/qd/selector/uniform.hpp>

#include <modules/nn2/mlp.hpp>
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "gen_dte.hpp" 

#include <cmath>
#include <algorithm>

#include <cstdlib>

#include "simu_fit_hexa_control_nn.hpp"
#include "best_fit_nn.hpp"

using namespace sferes;
using namespace sferes::gen::evo_float;
using namespace sferes::gen::dnn;

struct Params {
    struct nov {
        SFERES_CONST size_t deep = 3;
        SFERES_CONST double l = 0.1; // according to hand tuning made on the 2D arm simulation
        SFERES_CONST double k = 15; // TODO right value?
        SFERES_CONST double eps = 0.1;// TODO right value??
    };
  
    // TODO: move to a qd::
    struct pop {
        // number of initial random points
        SFERES_CONST size_t init_size = 1000;
        // size of a batch
        SFERES_CONST size_t size = 200;
        SFERES_CONST size_t nb_gen = 5000;
        SFERES_CONST size_t dump_period = 100;
    };

    struct dnn {
        SFERES_CONST size_t nb_inputs = 12 + 2 + 3; //previous commands(12) -> DOF3 = -DOF2 / distance to target / orientation
        SFERES_CONST size_t nb_outputs  = 12; //new commands
        SFERES_CONST size_t min_nb_neurons  = 41;
        SFERES_CONST size_t max_nb_neurons  = 100;
        SFERES_CONST size_t min_nb_conns  = 150;
        SFERES_CONST size_t max_nb_conns  = 300;
        SFERES_CONST float  max_weight  = 2.0f;
        SFERES_CONST float  max_bias  = 2.0f;

        SFERES_CONST float m_rate_add_conn  = 1.0f;
        SFERES_CONST float m_rate_del_conn  = 1.0f;
        SFERES_CONST float m_rate_change_conn = 1.0f;
        SFERES_CONST float m_rate_add_neuron  = 1.0f;
        SFERES_CONST float m_rate_del_neuron  = 1.0f;

        SFERES_CONST int io_param_evolving = true;
        //SFERES_CONST init_t init = random_topology;
        SFERES_CONST init_t init = ff;
    };

    struct parameters {
      SFERES_CONST float min = 0.0;
      SFERES_CONST float max = 1.0;
    };
    struct evo_float {
        SFERES_CONST float cross_rate = 0.1f;
        SFERES_CONST float mutation_rate = 0.03f;
        SFERES_CONST float eta_m = 10.0f;
        SFERES_CONST float eta_c = 10.0f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
    };
    struct qd {
        SFERES_CONST size_t dim = 2;
        SFERES_CONST size_t behav_dim = 3; //zones + target
        SFERES_ARRAY(size_t, grid_shape, 100, 100);
    };
};

template<typename fit_t, typename Model>
void visualise_behaviour(Eigen::Vector3d& target, Model& model){

  fit_t fit;
  fit.simulate(target, model);
}




int main(int argc, char **argv) 
{
    tbb::task_scheduler_init init(48);

    load_and_init_robot();

    
    using namespace sferes;
    using namespace nn;

    typedef Fit_hexa_control_nn<Params> fit_t;

    // typedef gen::EvoFloat<36, Params> gen_t;
    typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;

    typedef PfWSum<weight_t> pf_t;
    typedef AfSigmoidNoBias<> af_t;
    typedef sferes::gen::DnnFF<Neuron<pf_t, af_t>, Connection<weight_t>, Params> gen_t;

    typedef phen::Dnn<gen_t, fit_t, Params> phen_t;
    
	typedef boost::archive::binary_iarchive ia_t;

    phen_t model; 

    //Eigen::Vector3d target = {-0.211234, 0.59688,0.0};
    
    Eigen::Vector3d target = {-1.0, 1.0 ,0.0};
    const std::string filename = "model_25000_3.bin";

    std::cout << "model...loading" << std::endl;
        {
        std::ifstream ifs(filename , std::ios::binary);
        ia_t ia(ifs);
        ia >> model;
        }
  
    std::cout << "model...loaded" << std::endl;
    model.develop();
    std::cout << "model developed" << std::endl;
    std::cout << "model initialized" << std::endl;

	visualise_behaviour<fit_t>(target, model);

	global::global_robot.reset();


    return 0;
}
