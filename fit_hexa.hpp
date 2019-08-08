#ifndef ___FIT_HEXA_HPP__
#define ___FIT_HEXA_HPP__

#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/control/hexa_control.hpp>

#ifdef GRAPHIC
#include <robot_dart/graphics/graphics.hpp>
#endif


#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/constraint/ConstraintSolver.hpp>

#include "desc_hexa.hpp"

namespace global{
  std::shared_ptr<robot_dart::Robot> global_robot;
}

void load_and_init_robot()
{
  std::cout<<"INIT Robot"<<std::endl;
  global::global_robot = std::make_shared<robot_dart::Robot>("exp/example_dart_exp/ressources/hexapod_v2.urdf");
  global::global_robot->set_position_enforced(true);
  //global::global_robot->set_position_enforced(true);
  //global_robot->skeleton()->setPosition(1,100* M_PI / 2.0);
  
  global::global_robot->set_actuator_types(dart::dynamics::Joint::SERVO);
  global::global_robot->skeleton()->enableSelfCollisionCheck();
  std::cout<<"End init Robot"<<std::endl;
}



FIT_QD(Fit_hexa)
{
public:
  Fit_hexa(){  }
  
  typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;

  template<typename Indiv>
    void eval(Indiv& ind)
  {

    _ctrl.resize(36);
    for(size_t i=0;i<_ctrl.size();i++)
      {
	       _ctrl[i] = round( ind.data(i) * 1000.0 ) / 1000.0;// limite numerical issues
      }

    simulate(_ctrl);

    float activation=0;
    for(size_t i=0;i<_ctrl.size();i+=3)
      {
	activation+=_ctrl[i];
      }
    // the fitness is the average amplitude of each motor (corresponds to the average activation of the motors)
    this->_value = activation/12;

    
    // descriptor is the final position of the robot. 
    std::vector<double> desc;
    desc.push_back((_traj.back()[0]+1.5)/3);
    desc.push_back((_traj.back()[1]+1.5)/3);// this is re-centered and scaled, to be sure that the DB is between 0 and 1 (constraints coming from the grid container)
    
    this->set_desc(desc);

    if(desc[0]<0 || desc[0]>1 ||desc[1]<0 || desc[1]>1)
      this->_dead=true; //if something is wrong, we kill this solution. 
    
  }
  
  void simulate(const std::vector<double> ctrl) 
  {
    auto g_robot=global::global_robot->clone();
    g_robot->skeleton()->setPosition(5, 0.15);


     double ctrl_dt = 0.015;
     g_robot->add_controller(std::make_shared<robot_dart::control::HexaControl>(ctrl_dt, ctrl));
     std::static_pointer_cast<robot_dart::control::HexaControl>(g_robot->controllers()[0])->set_h_params(std::vector<double>(1, ctrl_dt));
    

     int n_c = 0;
     n_c = g_robot->num_controllers();
     std::cout << "nombre de controlleurs = " << n_c << std::endl;



    robot_dart::RobotDARTSimu simu(0.005);
#ifdef GRAPHIC
    simu.set_graphics(std::make_shared<robot_dart::graphics::Graphics>(simu.world()));
#endif

    simu.world()->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    simu.add_floor();
    simu.add_robot(g_robot);

    simu.add_descriptor(std::make_shared<robot_dart::descriptor::HexaDescriptor>(robot_dart::descriptor::HexaDescriptor(simu)));
  
    std::cout << "avant de run" << std::endl;
    simu.run(5);

    g_robot.reset();

    _traj=std::static_pointer_cast<robot_dart::descriptor::HexaDescriptor>(simu.descriptor(0))->traj;
  }


  

  
private:
  std::vector<double> _ctrl;
  std::vector<Eigen::VectorXf> _traj;

  
};



#endif
