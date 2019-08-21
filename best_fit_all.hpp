#ifndef BEST_FIT_ALL_
#define BEST_FIT_ALL_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "fit_behav.hpp"

namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitAll, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());


        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;

        //change it to depend from params 
        if (_cnt%Params::pop::dump_period == 0){ //for each dump period

          typedef boost::archive::binary_oarchive oa_t;

          std::cout << "writing...model" << std::endl;
          //const std::string fmodel = "/git/sferes2/exp/tmp/model_" + std::to_string(_cnt) + ".bin";
          const std::string fmodel = ea.res_dir() + "/model_" + std::to_string(_cnt) + ".bin";
      	  {
      	  std::ofstream ofs(fmodel, std::ios::binary);
                
      	  if (ofs.fail()){
      		  std::cout << "wolla ca s'ouvre pas" << std::endl;}  
      	
      	  oa_t oa(ofs);
          //oa << model;
          oa << *_best;
          }
        }

        if (_cnt == Params::pop::nb_gen - 1){
          test_and_save(ea);
        }

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

      template<typename E>
      void test_and_save(const E& ea){

        int cnt = 0;

        std::cout << "starting test and save" << std::endl;

        const std::string filename = ea.res_dir() + "/dict_models.txt";
        std::ofstream dict_file;
        dict_file.open(filename); 

        for( auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {

          std::vector<double> results(3);
          results = test_model(*it); //simulate and obtain fitness and behavior descriptors

          std::cout << "test unitaire - fitness: " << results[0] << " behavior descriptor: " << results[1] << " - " << results[2] << " - " << results[3] << std::endl;

          dict_file << "final_model_" + std::to_string(cnt) << "  " << results[0] << "  " << results[1] << "  " << results[2] << "  " << results[3] << "\n"; //save simulation results in dictionary file

          typedef boost::archive::binary_oarchive oa_t;
          const std::string fmodel = ea.res_dir() + "/final_model_" + std::to_string(cnt) + ".bin";
          {
          std::ofstream ofs(fmodel, std::ios::binary);
                
          if (ofs.fail()){
            std::cout << "wolla ca s'ouvre pas" << std::endl;}  
        
          oa_t oa(ofs);
          //oa << model;
          oa << *it;
          } //save model

          cnt ++;
        }

        dict_file.close();

        std::cout << std::to_string(cnt) + " models saved" << std::endl;
      }

      template <typename T>
      std::vector<double> test_model(T& model){

          //init variables
          double _vmax = 1;
          double _delta_t = 0.1;
          double _t_max = 10; //TMax guidé poto
          Eigen::Vector3d robot_angles;
          Eigen::Vector3d target;
          double dist = 0;

          Eigen::Vector3d prev_pos; //compute previous position
          Eigen::Vector3d output;
          Eigen::Vector3d pos_init;

          std::vector<double> zone_exp(3);
          std::vector<double> res(3);
          std::vector<double> results(4);

          robot_angles = {0,M_PI,M_PI}; //init everytime at the same place

          double radius;
          double theta;

          target = {-0.211234, 0.59688, 0.0};

          //get gripper's position
          prev_pos = forward_model(robot_angles);
          pos_init = forward_model(robot_angles);

          std::vector<float> inputs(5);

          model->develop();
          model->nn().init();

          //iterate through time
          for (int t=0; t< _t_max/_delta_t; ++t){
                
                inputs[0] = target[0] - prev_pos[0]; //get side distance to target
                inputs[1] = target[1] - prev_pos[1]; //get front distance to target
                inputs[2] = robot_angles[0];
                inputs[3] = robot_angles[1];
                inputs[4] = robot_angles[2];

                for (int j = 0; j < model->gen().get_depth()   + 1; ++j)
                  model->nn().step(inputs);
                
                for (int indx = 0; indx < 3; ++indx){
                  output[indx] = 2*(model->nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
                  robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
                }

                prev_pos = forward_model(robot_angles); //remplacer pour ne pas l'appeler deux fois

                res = get_zone(pos_init, target, prev_pos);
                zone_exp[0] = zone_exp[0] + res[0];
                zone_exp[1] = zone_exp[1] + res[1];
                zone_exp[2] = zone_exp[2] + res[2];

                target[2] = 0; //get rid of z coordinate
                prev_pos[2] = 0;

                if (sqrt(square(target.array() - prev_pos.array()).sum()) < 0.02){
                  dist -= sqrt(square(target.array() - prev_pos.array()).sum());
                 }

                else {
                  dist -= (log(1+t)) + (sqrt(square(target.array() - prev_pos.array()).sum()));
                }
              }

          Eigen::Vector3d final_pos; 
          final_pos = forward_model(robot_angles);

          if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){
            results[0] = 1.0 + dist/500; // -> 1
          }

          else {
            results[0] = dist/500; // -> 0
          }

          results[1] = zone_exp[0]/(_t_max/_delta_t);
          results[2] = zone_exp[1]/(_t_max/_delta_t); 
          results[3] = zone_exp[2]/(_t_max/_delta_t);

          return results;
      }

      Eigen::Vector3d forward_model(Eigen::VectorXd a){
    
        Eigen::VectorXd _l_arm=Eigen::VectorXd::Ones(a.size()+1);
        _l_arm(0)=0;
        _l_arm = _l_arm/_l_arm.sum();

        Eigen::Matrix4d mat=Eigen::Matrix4d::Identity(4,4);

        for(size_t i=0;i<a.size();i++){

          Eigen::Matrix4d submat;
          submat<<cos(a(i)), -sin(a(i)), 0, _l_arm(i), sin(a(i)), cos(a(i)), 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
          mat=mat*submat;
        }
        
        Eigen::Matrix4d submat;
        submat<<1, 0, 0, _l_arm(a.size()), 0, 1, 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
        mat=mat*submat;
        Eigen::VectorXd v=mat*Eigen::Vector4d(0,0,0,1);

        return v.head(3);

        }

  std::vector<double> get_zone(Eigen::VectorXf start, Eigen::Vector3d target, Eigen::VectorXf pos){
      
      
      std::vector<double> desc_add (3);
      
      Eigen::Vector3d middle;
      middle[0] = (start[0]+target[0])/2;
      middle[1] = (start[1]+target[1])/2;
      middle[2] = 1;
      
      std::vector<double> distances (3);
      distances = {0,0,0};
      
//std::cout << "get zone 1" << std::endl;

      distances[0] = sqrt((start[0] - pos[0])*(start[0] - pos[0]) + (start[1] - pos[1])*(start[1] - pos[1]));

      distances[1] = sqrt((target[0] - pos[0])*(target[0] - pos[0]) + (target[1] - pos[1])*(target[1] - pos[1]));

      distances[2] = sqrt((middle[0] - start[0])*(middle[0] - start[0]) + (middle[1] - start[1])*(middle[1] - start[1])); 

      
      Eigen::Vector3d vO2_M_R0; //vector 02M in frame R0; (cf sketch on page 4)
      vO2_M_R0[0] = pos[0] - start[0];
      //vO2_M_R0[0] = pos[0];
      vO2_M_R0[1] = pos[1] - start[1];
      //vO2_M_R0[1] = pos[1];
      vO2_M_R0[2] = 1;
    
      Eigen::Vector3d vMid_M_R0; //vector Middle_M in frame R0;
      vMid_M_R0[0] = pos[0] - middle[0];
      vMid_M_R0[1] = pos[1] - middle[1];
      vMid_M_R0[2] = 1;
      
      //Eigen::Matrix3d T; //translation matrix
      //T << 1,0,-start[0],0,1,-start[1],0,0,1; //translation matrix
      
      Eigen::Vector3d vO2_T;
      vO2_T[0] = target[0] - start[0];
      vO2_T[1] = target[1] - start[1];
      vO2_T[2] = 1;

      double theta = atan2(vO2_T[1], vO2_T[0]) - atan2(1, 0);
      
      if (theta > M_PI){
          theta -= 2*M_PI;
      }
      else if (theta <= -M_PI){
          theta += 2*M_PI;
      }
      
      Eigen::Matrix3d R;
      R << cos(theta), sin(theta), 0, -sin(theta), cos(theta), 0, 0, 0, 1; //rotation matrix
      
      Eigen::Vector3d vO2_M_R1; //vector 02M in frame R1;
      //vO2_M_R1 = T*vO2_M_R0;  
      vO2_M_R1 = R*vO2_M_R0;
    
      Eigen::Vector3d vMid_M_R1; //vector Middle_M in frame R1;
      vMid_M_R1 = R*vMid_M_R0;
      
      
      if (vO2_M_R1[0] < 0){ //negative zone (cf sketch on page 3)
          if (distances[0] < 0.1 || distances[1] < 0.1 || (abs(vMid_M_R1[0]) < 0.1 && abs(vMid_M_R1[1]) < distances[2])) {
              return {-1, 0, 0};
          }
          if ((distances[0] < 0.2 || distances[1] < 0.2 || (abs(vMid_M_R1[0]) < 0.2 && abs(vMid_M_R1[1]) < distances[2])) && (distances[0] >= 0.1 || distances[1] >= 0.1 || (abs(vMid_M_R1[0]) >= 0.1 && abs(vMid_M_R1[1]) < distances[2]))){
              return {0, -1, 0};
          }
          else {
              return {0,0,-1};
          }
      }
      
      else{ //positive zone
          if (distances[0] < 0.1 || distances[1] < 0.1 || (abs(vMid_M_R1[0]) < 0.1 && abs(vMid_M_R1[1]) < distances[2])) {
              return {1, 0, 0};
          }
          if ((distances[0] < 0.2 || distances[1] < 0.2 || (abs(vMid_M_R1[0]) < 0.2 && abs(vMid_M_R1[1]) < distances[2])) && (distances[0] >= 0.1 || distances[1] >= 0.1 || (abs(vMid_M_R1[0]) >= 0.1 && abs(vMid_M_R1[1]) < distances[2]))){
              return {0, 1, 0};
          }
          else {
              return {0,0,1};
          }
      }
  }

    protected:
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
      int _nbest = 3;
    };
  }
}
#endif
