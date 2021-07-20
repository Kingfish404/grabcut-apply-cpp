cmake_minimum_required(VERSION 3.10)
project(cg_apply)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /usr/local/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

include_directories(
        /usr/local/include
        /usr/local/include/opencv4
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(cg_grabcut src/main.cpp src/grabcut.hpp)
add_executable(cg_grabcut_ref src/grabcut_ref.cpp)

target_link_libraries(cg_grabcut /usr/local/lib)
target_link_libraries(cg_grabcut ${OpenCV_LIBRARIES})

target_link_libraries(cg_grabcut_ref /usr/local/lib)
target_link_libraries(cg_grabcut_ref ${OpenCV_LIBRARIES})
