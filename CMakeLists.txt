cmake_minimum_required(VERSION 2.8.3)

project(approxik)

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/deps/install)
set(deps_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/deps/install/lib;${CMAKE_SOURCE_DIR}/deps/install/lib/x86_64-linux-gnu")

# set the default build type
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/FindEigen3.cmake)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

set(CXXDEFINITIONS "-DNDEBUG")

if(ROCK) # Build inside robot construction kit (rock)?
    include(FindPkgConfig)

    pkg_check_modules(urdfdom_headers_approxik REQUIRED urdfdom_headers)
    pkg_check_modules(urdfdom REQUIRED urdfdom)
    pkg_check_modules(console_bridge REQUIRED console_bridge)
    pkg_check_modules(orocos_kdl REQUIRED orocos_kdl)
    pkg_check_modules(kdl_parser REQUIRED kdl_parser)

    set(CXXDEFINITIONS "${CXXDEFINITIONS} -DROCK")
else()   # You must install dependencies manually!
    find_package(urdfdom_headers_approxik REQUIRED)
    find_package(urdfdom REQUIRED)
    find_package(console_bridge REQUIRED)
    find_package(orocos_kdl REQUIRED)
    find_package(kdl_parser REQUIRED)
endif()

include_directories(${urdfdom_headers_INCLUDE_DIRS})
include_directories(${urdfdom_INCLUDE_DIRS})
link_directories(${urdfdom_LIBRARY_DIRS})
include_directories(${console_bridge_INCLUDE_DIRS})
link_directories(${console_bridge_LIBRARY_DIRS})
include_directories(${orocos_kdl_INCLUDE_DIRS})
link_directories(${orocos_kdl_LIBRARY_DIRS})
include_directories(${kdl_parser_INCLUDE_DIRS})
link_directories(${kdl_parser_LIBRARY_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

find_library(KDL_LIBRARY REQUIRED NAMES orocos-kdl HINTS ${orocos_kdl_LIBRARY_DIRS})

add_definitions(${CXXDEFINITIONS})

# check existence of c++11 compiler
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPILER_SUPPORTS_OPENMP)
    if(COMPILER_SUPPORTS_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(openmp_LIBRARIES "gomp")
    else()
        set(openmp_LIBRARIES "")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    CHECK_CXX_COMPILER_FLAG("-msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2" COMPILER_SUPPORTS_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2")
endif()

include_directories(.)

# TODO remove hack as soon as CppNumericalSolver can be installed
set(cppoptlib_LIBRARIES pthread)
set(cppoptlib_LIBRARY_DIRS "")
set(cppoptlib_INCLUDE_DIRS deps/CppNumericalSolvers/include)
include_directories(${cppoptlib_INCLUDE_DIRS})

add_executable(approxik src/approxik.cpp)
target_link_libraries(approxik
    ${orocos_kdl_LIBRARIES} ${urdfdom_LIBRARIES} ${kdl_parser_LIBRARIES}
    ${cppoptlib_LIBRARIES} ${openmp_LIBRARIES})

configure_file(
    ${CMAKE_SOURCE_DIR}/python/config.py.in
    ${CMAKE_SOURCE_DIR}/python/config.py)
# TODO install python wrapper

install(FILES approxik/approxik.hpp DESTINATION include/${PROJECT_NAME})

set(PKG_CONFIG_CFLAGS -I${CMAKE_INSTALL_PREFIX}/include)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.pc.in
               ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.pc)
install(FILES ${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)
