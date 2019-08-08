#! /usr/bin/env python
import sys
import os
import sferes
sys.path.insert(0, sys.path[0]+'/waf_tools')
print sys.path[0]


from waflib.Configure import conf

import dart
import robot_dart

def options(opt):
    opt.load('dart')
    opt.load('robot_dart')

@conf
def configure(conf):
    print 'conf exp:'
    conf.load('dart')
    conf.check_dart()
    conf.load('robot_dart')
    conf.check_robot_dart()
    
    print 'done'
    
def build(bld):

    bld.program(features = 'cxx',
                source = 'dart_exp.cpp',
                includes = '. ../../',
                uselib = 'ROBOTDART TBB BOOST EIGEN PTHREAD MPI DART ROBOT_DART',
                use = 'sferes2',
                target = 'example')
    bld.program(features = 'cxx',
                source = 'dart_exp.cpp',
                includes = '. ../../',
                uselib = 'ROBOTDART TBB BOOST EIGEN PTHREAD MPI DART ROBOT_DART DART_GRAPHIC',
                use = 'sferes2',
                defines = ["GRAPHIC"],
                target = 'example_graphic')

    bld.program(features = 'cxx',
                source = 'test_dart.cpp',
                includes = '. ../../',
                uselib = 'ROBOTDART TBB BOOST EIGEN PTHREAD MPI DART ROBOT_DART',
                use = 'sferes2',
                target = 'test_dart')


    bld.program(features = 'cxx',
                source = 'test_dart.cpp',
                includes = '. ../../',
                uselib = 'ROBOTDART TBB BOOST EIGEN PTHREAD MPI DART ROBOT_DART DART_GRAPHIC',
                use = 'sferes2',
                defines = ["GRAPHIC"],
                target = 'test_dart_graphic')

                            
