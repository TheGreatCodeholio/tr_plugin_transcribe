cmake_minimum_required(VERSION 3.13)
project(tr-plugin-transcriber VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED)   # headers are enough for BOOST_LOG_TRIVIAL used below
find_package(CURL REQUIRED)

add_library(transcriber
        MODULE
        src/transcriber_plugin.cc
)

include_directories(
        ${CMAKE_BINARY_DIR}/../                 # /src/trunk-recorder (TR tree root)
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # this plugin's headers
)

target_link_libraries(transcriber
        PRIVATE
        CURL::libcurl
        trunk_recorder_library
)

# GCC < 9 only
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(transcriber PRIVATE stdc++fs)
endif()

set_target_properties(transcriber PROPERTIES
        OUTPUT_NAME "transcriber"
        PREFIX "lib"
)

install(TARGETS transcriber LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/trunk-recorder)
