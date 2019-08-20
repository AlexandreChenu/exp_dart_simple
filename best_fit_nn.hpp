#ifndef BEST_FIT_NN_
#define BEST_FIT_NN_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "fit_hexa_control_nn.hpp"

namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitNN, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());


        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;

        //change it to depend from params 
        if (_cnt%Params::pop::dump_period == 0){ //save model

          typedef boost::archive::binary_oarchive oa_t;

          std::cout << "writing...model" << std::endl;
          //const std::string fmodel = std::string("/git/sferes2/exp/tmp/model_") + std::to_string(_cnt) + ".bin";
          const std::string fmodel = ea.res_dir() + "/model_" + std::to_string(_cnt) + ".bin";
          {
          std::ofstream ofs(fmodel, std::ios::binary);
          oa_t oa(ofs);
          //oa << model;
          oa << *_best;
          }
          std::cout << "model written" << std::endl;}
        _cnt += 1;
      }

      void show(std::ostream& os, size_t k) {
        _best->develop();
        _best->show(os);
        _best->fit().set_mode(fit::mode::view);
        _best->fit().eval(*_best);
      }
      const boost::shared_ptr<Phen> best() const {
        return _best;
      }
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(_best);
      }

    protected:
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
    };
  }
}
#endif
