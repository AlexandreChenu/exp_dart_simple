#!/bin/bash

./waf configure --exp exp_dart_simple --dart /workspace --kdtree /workspace/include --robot_dart /workspace
./waf --exp exp_dart_simple
