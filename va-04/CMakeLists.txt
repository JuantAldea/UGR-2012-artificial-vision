project(va-04)
cmake_minimum_required(VERSION 2.8)
add_executable(${CMAKE_PROJECT_NAME}  main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi -pedantic-errors -Wall -Wextra")
find_package(OpenCV REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
#set(CMAKE_BUILD_TYPE Debug)

