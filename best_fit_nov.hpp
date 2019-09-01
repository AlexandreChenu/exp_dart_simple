#ifndef BEST_FIT_NOV_
#define BEST_FIT_NOV_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "fit_hexa_control_nn.hpp"

#include <robot_dart/robot_dart_simu.hpp>
//#include <robot_dart/control/hexa_control.hpp>
#include "hexa_control.hpp"


#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>

#include <modules/nn2/mlp.hpp>
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>

#include <modules/nn2/gen_dnn_ff.hpp>

#include "desc_hexa.hpp"


namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitNov, Stat) {
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


          const std::string fmodel = ea.res_dir() + "/model_" + std::to_string(_cnt) + ".bin";
          
	  Eigen::MatrixXd zones_cnt = Eigen::MatrixXd::Zero(101,101);
          Eigen::Vector3d target;

          target = {-0.5, 0.5,0.0};

          std::cout << "pop size: " << ea.pop().size() << std::endl;

          for (int i = 0; i < ea.pop().size(); ++i){
                zones_cnt += run_simu(*ea.pop()[i], target);
                }

	  int sum_zones = 0;

          for (float i = 0; i < 101; i+=1){
            for (float j = 0; j < 101; j+=1){
              if (zones_cnt(i,j) != 0)
                sum_zones += 1;
              }}
	 
          double novelty_score = sum_zones;
          double novelty_score_n = novelty_score /(100*100);

	  std::cout << "novelty score is: " << novelty_score << std::endl;

          std::cout << "normalized novelty score is: " << novelty_score_n <<  std::endl;

          _nov_scores.push_back(novelty_score);
	  }

        _cnt += 1;

	if (_cnt == Params::pop::nb_gen){

          std::cout << "Saving novelty scores" << std::endl;

          std::string filename_out = ea.res_dir() + "novelty_gte.txt"; //file containing samples
          //std::string filename_out = "/git/sferes2/results_signb_nov/novelty_gte.txt";
          std::ofstream out_file; 
          out_file.open(filename_out);

          if (!out_file) { //quick check to see if the file is open
            std::cout << "Unable to open file " << filename_out;
            exit(1);}   // call system to stop

          for (int i = 0; i < _nov_scores.size(); i++){
            out_file << _nov_scores[i] << std::endl;
          }
          out_file.close();
        }

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
	
      template <typename T>
      Eigen::MatrixXd run_simu(T & model, Eigen::Vector3d target) { 
	
     	simulate(target,model);	 
	
	int size = _traj.size();
	Eigen::MatrixXd work_zones_cnt = Eigen::MatrixXd::Zero(101,101);

	for (int i = 0; i < size; i++){

	      int x_int = _traj[i][0]*100;
              int y_int = _traj[i][1]*100;

              int indx_X =0;
              int indx_Y =0;
              
              if (x_int %2 !=0)
                  indx_X = (x_int + 100)/2;
              
              else 
                  indx_X = (x_int + 101)/2 ;
              
              if (y_int %2 !=0)
                  indx_Y = (y_int + 100)/2;
              
              else 
                  indx_Y = (y_int + 101)/2;
          
              work_zones_cnt(indx_X,indx_Y) ++;}

	if (sqrt((target[0] - _traj[-1][0])*(target[0] - _traj[-1][0]) + (target[1] - _traj[-1][1])*(target[1] - _traj[-1][1]) < 0.05){
			std::cout << "task successful" << std::endl;
			return work_zones_cnt;}
	else{
		return Eigen::MatrixXd::Zero(101,101);}
    }

  template<typename Model>
  void simulate(Eigen::Vector3d& target, Model& model){

    auto g_robot=global::global_robot->clone();
    g_robot->skeleton()->setPosition(5, 0.15);


    double ctrl_dt = 0.015;
    g_robot->add_controller(std::make_shared<robot_dart::control::HexaControlNN<Model>>());
    //std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->set_h_params(std::vector<double>(1, ctrl_dt));

    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->setModel(model); //TODO : understand why do we use a static pointer cast

    std::static_pointer_cast<robot_dart::control::HexaControlNN<Model>>(g_robot->controllers()[0])->setTarget(target);

    robot_dart::RobotDARTSimu simu(0.005); //creation d'une simulation

    simu.world()->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    simu.add_floor();
    simu.add_robot(g_robot);

    simu.add_descriptor(std::make_shared<robot_dart::descriptor::HexaDescriptor>(robot_dart::descriptor::HexaDescriptor(simu)));
    simu.add_descriptor(std::make_shared<robot_dart::descriptor::DutyCycle>(robot_dart::descriptor::DutyCycle(simu)));

    simu.run(7);

    _traj = std::static_pointer_cast<robot_dart::descriptor::HexaDescriptor>(simu.descriptor(0))->traj;
    g_robot.reset();}


    protected:
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
      std::vector<Eigen::VectorXf> _traj;
      std::vector<double> _nov_scores;
	    
    };
  }
}
#endif
