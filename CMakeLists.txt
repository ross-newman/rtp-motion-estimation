
cmake_minimum_required(VERSION 2.8)
project(motion-est)

# setup CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++0x -lz")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# Qt is used as mutex in gstreamer camera (installed by ubuntu-desktop)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# setup CUDA
find_package(CUDA)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 -gencode arch=compute_53,code=sm_53
)

# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/bin)
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)
# ${CMAKE_SYSTEM_PROCESSOR}
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/${CMAKE_SYSTEM_PROCESSOR})

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/${CMAKE_SYSTEM_PROCESSOR}/linux/release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# build C/C++ interface
include_directories(${PROJECT_INCLUDE_DIR})
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

# copy network data
file(GLOB videoData ${PROJECT_SOURCE_DIR}/data/*)

foreach(include ${videoData})
	message("-- Copying ${include}")
	file(COPY ${include} DESTINATION ${PROJECT_OUTPUT_DIR}/data)
	#configure_file(${include} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()

# build samples & tools
add_subdirectory(demos/motion_estimation)

