cmake_minimum_required(VERSION 3.17)
project(GpuLabProject)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)


find_package(Boost COMPONENTS system filesystem REQUIRED)
if (WIN32)
    set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
    set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

    include_directories(${BOOST_INC})
    link_directories(${BOOST_LIB})
elseif(UNIX)
    include_directories(${Boost_INCLUDE_DIR})
endif()

if (WIN32)
    file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

include_directories(Core OpenCL)

add_custom_target(disparity_cl COMMAND
        ${CMAKE_COMMAND}
        -DXIF=${CMAKE_CURRENT_SOURCE_DIR}/src/disparity.cl
        -DXOF=${CMAKE_CURRENT_SOURCE_DIR}/src/gen/disparity.cl
        -P ${CMAKE_CURRENT_SOURCE_DIR}/disparity_cl.cmake)

file(GLOB_RECURSE f_src "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE f_core "${CMAKE_SOURCE_DIR}/Core/*.cpp" "${CMAKE_SOURCE_DIR}/Core/*.c")
file(GLOB_RECURSE f_ocl "${CMAKE_SOURCE_DIR}/OpenCL/*.cpp" "${CMAKE_SOURCE_DIR}/OpenCL/*.c")

include_directories(${Boost_INCLUDE_DIR})

#add_library(Core STATIC ${f_core})
#add_library(OpenCL STATIC ${f_ocl})

#add_executable(disparity ${f_src})
#target_link_libraries(disparity Core OpenCL ${OpenCL_LIBRARY})
add_executable(disparity ${f_core} ${f_ocl} ${f_src})
add_dependencies(disparity disparity_cl)
target_include_directories(disparity PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} Core OpenCL)
target_link_libraries(disparity ${OpenCL_LIBRARY} dl)
