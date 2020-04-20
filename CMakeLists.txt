project( hello_face )

cmake_minimum_required( VERSION 3.7 )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED True )
set( OpenCV_DIR /opt/opencv-4.2.0/lib/cmake/opencv4 )

find_package( OpenCV REQUIRED )

set( TI_OCL_INCLUDE_DIR "/usr/share/ti/opencl" )
set( TIDL_API_INCLUDE_DIR "/usr/share/ti/tidl/tidl_api/inc" )
set( GCC_COVERAGE_LINK_FLAGS "-lTIOpenCL -locl_util" )
set( CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS} )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb" )

#add_library( tidl SHARED IMPORTED )
#set_property( TARGET tidl PROPERTY IMPORTED_LOCATION "/usr/share/ti/tidl/tidl_api/tidl.so" )

add_library( tidl_api STATIC IMPORTED )
set_property( TARGET tidl_api PROPERTY IMPORTED_LOCATION "/usr/share/ti/tidl/tidl_api/tidl_api.a" )

add_library( tidl_imgutil STATIC IMPORTED )
set_property( TARGET tidl_imgutil PROPERTY IMPORTED_LOCATION "/usr/share/ti/tidl/tidl_api/tidl_imgutil.a" )

add_executable( video_test video_test.cpp )
target_link_libraries( video_test ${OpenCV_LIBS} )

add_executable( face_detect face_detect.cpp )
target_include_directories( face_detect PRIVATE ${TI_OCL_INCLUDE_DIR} ${TIDL_API_INCLUDE_DIR} )
target_link_libraries( face_detect PUBLIC ${OpenCV_LIBS} tidl_api tidl_imgutil )
