# Copyright 2018, alex at staticlibs.net
# Copyright 2018, mike at myasnikov.mike@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required ( VERSION 2.8.12 )

project ( patterns_compare CXX )

# dependencies
#set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "" )
#set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "" )
#set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "" )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++1y -g")

set(SOURCES
    src/main.cpp
    src/points_processor.cpp
    src/histogram_calculate.cpp
)

#set ( LIBPATH  "/usr/local/lib" )
set ( LIBPATH  "/home/mike/workspace/utils/opencv/debug_modules/lib")


 set(LIBS
    ${LIBPATH}/libopencv_core.so
    ${LIBPATH}/libopencv_imgproc.so
    ${LIBPATH}/libopencv_objdetect.so
    ${LIBPATH}/libopencv_highgui.so
#    ${LIBPATH}/libopencv_contrib.so
    ${LIBPATH}/libopencv_features2d.so
    ${LIBPATH}/libopencv_imgcodecs.so
    ${LIBPATH}/libopencv_face.so
    ${LIBPATH}/libopencv_ml.so
    ${LIBPATH}/libopencv_calib3d.so

    pthread
    /home/mike/workspace/utils/xlnt/build/source/libxlnt.so.1.2
#    opencv_imgproc
    # ssl
    # crypto
 )

# target_include_directories ( ${PROJECT_NAME} BEFORE PRIVATE
#         $ENV{WILTON_HOME}/devel/include
# )

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories ( ${PROJECT_NAME} BEFORE PRIVATE
     /usr/local/include/opencv4
)

target_link_libraries ( ${PROJECT_NAME} ${LIBS})
