#ifndef ROBOT_DART_DESCRIPTOR_HEXA_HPP
#define ROBOT_DART_DESCRIPTOR_HEXA_HPP

// for size_t
#include <cstddef>
#include <cstdlib>

namespace robot_dart {
    class RobotDARTSimu;
    class Robot;

    namespace descriptor {
	
//HexaDescriptor allows access to robots positions and angles
      struct HexaDescriptor:public BaseDescriptor{ 
        public:
	HexaDescriptor(RobotDARTSimu& simu, size_t desc_dump = 1):BaseDescriptor(simu,desc_dump),_on_back(false)
	{}
	std::vector<Eigen::VectorXf> traj;
	virtual void operator()()
	{
	  auto pos=_simu.robots().back()->skeleton()->getPositions().head(6).tail(3).cast <float> ();
	  traj.push_back(pos.head(2));
	  auto ang = _simu.robots().back()->skeleton()->getPositions().head(6).head(3).cast <float> ();
	  
	  if (abs(ang[1]) > 1.6)
		_on_back = true;
	}
	
	bool on_back(){return _on_back;} 
		
	private :
		bool _on_back; //true if robot is on its back
	      
      }; //struct HexaDescriptor

     struct DutyCycle:public BaseDescriptor{
      public:
	DutyCycle(RobotDARTSimu& simu, size_t desc_dump = 1):BaseDescriptor(simu,desc_dump),_body_contact(false)
	{}

	virtual void operator()()
	{
	const dart::collision::CollisionResult& col_res = _simu.world()->getLastCollisionResult();	  
	_body_contact = false;    
 
   	  dart::dynamics::BodyNodePtr part_to_check = _simu.robots().back()->skeleton()->getBodyNode("base_link");

	  if(col_res.inCollision(part_to_check)){
		_body_contact=true;
		std::cout << "KOKO" << std::endl;
	      }
	  if (_body_contact)
		_collision = true;
	  
	} //void operator

	bool body_contact(){return _collision;}

      protected:
	bool _body_contact; //assess if robot body suffers collision
	bool _collision = false;	
      };//struct dutycycle


    } // namespace descriptor
} // namespace robot_dart

#endif
