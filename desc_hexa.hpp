#ifndef ROBOT_DART_DESCRIPTOR_HEXA_HPP
#define ROBOT_DART_DESCRIPTOR_HEXA_HPP

// for size_t
#include <cstddef>

namespace robot_dart {
    class RobotDARTSimu;
    class Robot;

    namespace descriptor {

      struct HexaDescriptor:public BaseDescriptor{
        public:
	HexaDescriptor(RobotDARTSimu& simu, size_t desc_dump = 1):BaseDescriptor(simu,desc_dump)
	{}
	std::vector<Eigen::VectorXf> traj;
	virtual void operator()()
	{
	  auto pos=_simu.robots().back()->skeleton()->getPositions().head(6).tail(3).cast <float> ();
	  traj.push_back(pos.head(2));
	  //std::cout<<pos.head(2).transpose()<<std::endl;
	  //if( traj.back()[0] > 1 || traj.back()[1] > 1 || traj.back()[0] < -1 || traj.back()[1] < -1 )
	  //std::cout<<"ERROR "<<traj.back().transpose()<<std::endl;
	}
      }; //struct HexaDescriptor

     struct DutyCycle:public BaseDescriptor{
      public:
	DutyCycle(RobotDARTSimu& simu, size_t desc_dump = 1):BaseDescriptor(simu,desc_dump),_body_contact(false)
	{}

	virtual void operator()()
	{
	  const dart::collision::CollisionResult& col_res = _simu.world()->getLastCollisionResult();


	  std::string part_name;
	  part_name = "body";

      dart::dynamics::BodyNodePtr part_to_check = _simu.robots().back()->skeleton()->getBodyNode(part_name);
      if(col_res.inCollision(part_to_check))
        _body_contact=true;
	  
	} //void operator

	bool body_contact(){return _body_contact;}

      protected:
	bool _body_contact;
	
      };//struct dutycycle


    } // namespace descriptor
} // namespace robot_dart

#endif
