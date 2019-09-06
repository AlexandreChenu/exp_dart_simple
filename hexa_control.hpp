#ifndef ROBOT_DART_CONTROL_HEXA_CONTROL_NN
#define ROBOT_DART_CONTROL_HEXA_CONTROL_NN

#include "hexapod_controller_simple.hpp"
#include "policy_control.hpp"

#include <cmath>
#include <modules/nn2/mlp.hpp>
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>

namespace robot_dart {
    namespace control {

        template <typename Indiv>
        struct HexaPolicyNN {
        public:
            void set_params(const std::vector<double>& ctrl)
            {
	    
            }

            size_t output_size() const { return 18; }

            Eigen::VectorXd query(const std::shared_ptr<robot_dart::Robot>& robot, double t)
            {
		if (!_h_params_set) {
                    _dt = robot->skeleton()->getTimeStep();
                }
                auto angles = get_angles(robot, t);
		
		Eigen::VectorXd target_positions = Eigen::VectorXd::Zero(18 + 6);

		for(size_t i = 0; i < angles.size(); i+=3){

			target_positions(i + 6 + 0) = angles[i + 0] * M_PI_4 / 2;
			target_positions(i + 6 + 1) = angles[i + 1] * M_PI_4;
			target_positions(i + 6 + 2) = angles[i + 2] * M_PI_4;
		}

                Eigen::VectorXd q = robot->skeleton()->getPositions();
                Eigen::VectorXd q_err = target_positions - q;

                double gain = 1.0 / (dart::math::constants<double>::pi() * _dt);
                Eigen::VectorXd vel = q_err * gain;

                return vel.tail(18);
            }

            void set_h_params(const std::vector<double>& h_params)
            {
                _dt = h_params[0];
                _h_params_set = true;
            }

            std::vector<double> h_params() const
            {
                return std::vector<double>(1, _dt);
            }

            //template <typename Indiv>
            void setModel (Indiv & model)
            {
                _model = model;
            }

            void setTarget (Eigen::Vector3d & target)
            {
                _target = target;
            }

            std::vector<double> get_angles(const std::shared_ptr<robot_dart::Robot>& robot, double t)
            {
		 double p_max = 1.0;
                int n_Dof = 12;

                std::vector<float> inputs(2 + n_Dof + 3 + 1); //nb of inputs correspond to NN inputs

                //auto pos?
                auto pos= robot->skeleton()->getPositions().head(6).tail(3).cast <float> (); //obtain robot's position
		

		inputs[0] = pos[0] - _target[0]; //inputs is gradient of position
                inputs[1] = pos[1] - _target[1];
		
		Eigen::VectorXd prev_commands_full = robot->skeleton()->getPositions();
		
                
                std::vector<double> prev_commands;
                for (int i = 1; i < 19; i++){ 
                    if (i % 3 != 0 ){
                        prev_commands.push_back(prev_commands_full[5 + i]);} 
                }
		  

                for (int i = 0; i < n_Dof ; i++){
                    inputs[2 + i] = prev_commands[i]; 
                }

                //auto angles? 
                auto angles = robot->skeleton()->getPositions().head(6).head(3).cast <float> (); //obtain robot's orientation
		
                for (int i = 0; i < 3 ; i++){
                    inputs[2 + n_Dof + i] = angles[i]; // yaw / pitch / roll
                }

		inputs[2 + n_Dof + 3] = std::fmod(t,1);

                _model.gen().init();

		for (int j = 0; j < 10 + 1; ++j)
                    _model.gen().step(inputs); 

                Eigen::VectorXd out_nn(12);
                for (int indx = 0; indx < n_Dof; indx ++ ){
                    out_nn[indx] = 2*(_model.nn().get_outf(indx) - 0.5)*p_max; //TODO : check if p_max is well adjusted - mapping betwenn -p_max:p_max with sigmoid
                }

                std::vector<double> commands_out;
                for (int i = 0; i < 12; i++){

                        if (i % 2 == 1){
                            commands_out.push_back(out_nn[i]);
                            commands_out.push_back(out_nn[i]); //add same value for DOF3
                        }
                        else{
                            commands_out.push_back(out_nn[i]);
                        }
                    }
		
                return commands_out;}

            

        protected:

            double _dt;
            bool _h_params_set = false;
            Indiv _model; 
            Eigen::Vector3d _target;


        };

        template<typename Model>
        using HexaControlNN = robot_dart::control::PolicyControl<HexaPolicyNN<Model>>;
    } // namespace control
} // namespace robot_dart

#endif
