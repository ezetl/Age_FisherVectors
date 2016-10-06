project(fvfaces)
cmake_minimum_required(VERSION 2.8.9)


set(LIBS "${CMAKE_CURRENT_SOURCE_DIR}/libs")
set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(DATA_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/data")

include(${LIBS}/dlib/cmake)

# Uncomment this line to enable drawing the faces on screen while they're processed
#add_definitions(-DDRAW_FACES)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
add_definitions ("-Wall")

MACRO(add_code name)
   add_definitions(-DDATA_FOLDER="${DATA_FOLDER}")
   message("Setting data dir to:  " ${DATA_FOLDER})

   execute_process(COMMAND grep ARM /proc/cpuinfo COMMAND head -1 OUTPUT_VARIABLE ARM)
   if(CMAKE_COMPILER_IS_GNUCXX)
       if (NOT (ARM STREQUAL ""))

          add_definitions(-DARM_ARCH)
          add_definitions(-DOPENCV_FACES)
          MESSAGE("Compiling for ARM architecture (OrangePi Plus 2).")
          set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
          set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
          set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -march=native -mcpu=cortex-a7 -mtune=cortex-a7 \
                               -mfpu=neon-vfpv4 -funsafe-math-optimizations -mfloat-abi=hard \
                               -fomit-frame-pointer -finline-small-functions -findirect-inlining \
                               -finline-functions -funroll-loops")

       else()

           set(DUSE_AVX_INSTRUCTIONS ON)
           MESSAGE("Compiling for native architecture.")
           set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

       endif()
   endif()

   # OpenCV
   find_package(OpenCV)
   if (OpenCV_FOUND)
       include_directories(${OpenCV_INCLUDE_DIRS})
   endif()

   # Use DLib implementations of these dependencies
   option(DLIB_USE_BLAS OFF)
   option(DLIB_USE_LAPACK OFF)

   # VLFeat
   FIND_PATH(VLFEAT_INCLUDE_DIR vl/generic.h
       PATHS
            ${LIBS}/vlfeat/
       DOC "Location of VLFEAT Headers"
       )
   FIND_LIBRARY(VLFEAT_LIBS
       NAMES vl
       PATHS ${LIBS}/vlfeat
       )
   MESSAGE("VLFEAT INCLUDES:  " ${VLFEAT_INCLUDE_DIR})
   MESSAGE("VLFEAT LIBS:  " ${VLFEAT_LIBS})

   INCLUDE_DIRECTORIES(${VLFEAT_INCLUDE_DIR})

   ADD_EXECUTABLE(${name} ${SRC}/${name}.cpp)

   TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib ${VLFEAT_LIBS})

ENDMACRO()

add_code(search)
add_code(rr)
