cmake_minimum_required(VERSION 3.17)
project(GpuLabProject)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(lib)

add_custom_target(disparity_cl COMMAND
        ${CMAKE_COMMAND}
        -DXIF=${CMAKE_CURRENT_SOURCE_DIR}/src/disparity.cl
        -DXOF=${CMAKE_CURRENT_SOURCE_DIR}/src/gen/disparity.cl
        -P ${CMAKE_CURRENT_SOURCE_DIR}/disparity_cl.cmake)

file(GLOB_RECURSE f_src "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB_RECURSE f_core "${CMAKE_SOURCE_DIR}/lib/Core/*.cpp" "${CMAKE_SOURCE_DIR}/lib/Core/*.c")
file(GLOB_RECURSE f_ocl "${CMAKE_SOURCE_DIR}/lib/OpenCL/*.cpp" "${CMAKE_SOURCE_DIR}/lib/OpenCL/*.c")

include_directories(${Boost_INCLUDE_DIR})

add_library(Core STATIC ${f_core})
add_library(OpenCL STATIC ${f_ocl})

add_executable(disparity ${f_src})
add_dependencies(disparity disparity_cl)
target_link_libraries(disparity Core OpenCL ${OpenCL_LIBRARY})
