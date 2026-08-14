cmake_minimum_required(VERSION 3.12)
project(pendulum_tracker)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# Cria o módulo compartilhado do Python (.so)
pybind11_add_module(pendulum_tracker release/source/tracker.cpp)
target_link_libraries(pendulum_tracker PRIVATE ${OpenCV_LIBS})