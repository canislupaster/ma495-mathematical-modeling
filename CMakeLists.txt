cmake_minimum_required(VERSION 3.17)
project(bobo CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_definitions(BUILD_DEBUG)
   add_compile_options(-fsanitize=address)
   add_link_options(-fsanitize=address)
endif ()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main main.cpp)
target_link_libraries(main Eigen3::Eigen)

add_executable(old old.cpp)
target_link_libraries(old Eigen3::Eigen)