cmake_minimum_required(VERSION 3.0)
project(PenduloTrack)

# Pede para o Linux achar o OpenCV instalado
find_package(OpenCV REQUIRED)

# Avisa onde estão os headers (.h)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(headers)

# Configura o C++ moderno (C++11 ou superior)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cria o executável final chamado "PenduloApp" juntando os 3 arquivos principais
add_executable(PenduloApp 
    release/source/main.cpp 
    release/source/calibration.cpp 
    release/source/offlineGraph.cpp
)

# Liga as bibliotecas necessárias (OpenCV e Threads)
target_link_libraries(PenduloApp ${OpenCV_LIBS} pthread)