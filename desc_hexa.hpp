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
      };
    } // namespace descriptor
} // namespace robot_dart

#endif
