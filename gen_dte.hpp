//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
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


// #include <sferes/gen/evo_float.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>
#include <iostream>
#include <cmath>

#ifndef DTE_HPP_
#define DTE_HPP_

namespace sferes{
  namespace gen{

    template<typename N, typename C, typename Params>
    class Dte : 
      //public sferes::gen::EvoFloat <Size, Params, Exact>, 
      public sferes::gen::DnnFF <N, C, Params> {
      //public EvoFloat <Size, Params, Exact> {
      

    public:
      typedef Params params_t;
      typedef Dte< N, C, Params> this_t; 

      Dte() {}

      //std::vector<double> _targ = {0,0,0};

      void init() {
        //std::cout << "START INIT" <<std::endl;
        sferes::gen::DnnFF<N, C, Params>::init();
        //std::cout << "END INIT" <<std::endl;
        }

      void random() {
        //std::cout << "START RANDOM" <<std::endl;
        sferes::gen::DnnFF<N, C, Params>::random();
        //std::cout << "DnnFF random done" <<std::endl;
        //BOOST_FOREACH(double &v, this->_targ) v = misc::rand<double>();
        double radius = ((double) rand() / (RAND_MAX)); //TODO: adapt to n dimensions
        double theta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
        //std::cout << "compute rad and theta" << std::endl;
        _targ[0] = radius*cos(theta);
        _targ[1] = radius*sin(theta);
        _targ[2] = 0;
        //std::cout << "END RANDOM" <<std::endl;
      }

      void cross(const sferes::gen::DnnFF<N, C, Params>& o, sferes::gen::DnnFF<N, C, Params>& c1, sferes::gen::DnnFF<N, C, Params>& c2) {
        //std::cout << "START CROSS" <<std::endl;
        sferes::gen::DnnFF<N, C, Params>::cross(o, c1, c2);
        //std::cout << "END CROSS" <<std::endl;
         //add cross for float
       }

      void mutate() {
         //sferes::gen::EvoFloat<Size, Params, Exact>::mutate();
        //std::cout << "START MUTATE" <<std::endl;
         sferes::gen::DnnFF<N, C, Params>::mutate();

         for (int i = 0; i < 2; i++){ //polynomial mutation for target 
          if (misc::rand<float>() < Params::evo_float::mutation_rate){
            SFERES_CONST float eta_m = Params::evo_float::eta_m;
            assert(eta_m != -1.0f);
            float ri = misc::rand<float>();
            float delta_i = ri < 0.5 ?
                          pow(2.0 * ri, 1.0 / (eta_m + 1.0)) - 1.0 :
                          1 - pow(2.0 * (1.0 - ri), 1.0 / (eta_m + 1.0));
            assert(!std::isnan(delta_i));
            assert(!std::isinf(delta_i));
            _targ[i] = _targ[i] + delta_i;
            misc::put_in_range(_targ[i], 0.0f, 1.0f);
          }
       }
        _targ[2] = 0;
       //std::cout << "END MUTATE" <<std::endl;
     }

     std::vector<double> get_target() const {

      return _targ; 
     }

    private: 
      
      double _rad = ((double) rand() / (RAND_MAX));
      double _teta = 2*M_PI*(((double) rand() / (RAND_MAX))-0.5);
      std::vector<double> _targ = {_rad*cos(_teta),_rad*sin(_teta),0};



    };
  }
}



#endif