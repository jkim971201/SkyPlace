# GPU Louvain CMAKE Compile Settings
project(Louvain LANGUAGES CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_C_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)

set(Louvain_HOME ${CMAKE_CURRENT_SOURCE_DIR})
set(Louvain_SRC 
utils.cu
community_aggregation.cu
modularity_optimisation.cu
removeSparse.cu
gpulouvain.cu)

set(Louvain_HEADERS gpulouvain.h)

enable_language("CUDA")

add_definitions(-DUSE_CUDA)

add_library(${PROJECT_NAME} STATIC ${Louvain_SRC})

set_target_properties(${PROJECT_NAME}
	PROPERTIES CUDA_ARCHITECTURES "75"
	ARCHIVE_OUTPUT_DIRECTORY ${Louvain_HOME}/lib
	LIBRARY_OUTPUT_DIRECTORY ${Louvain_HOME}/lib
	RUNTIME_OUTPUT_DIRECTORY ${Louvain_HOME}/lib
)

find_package(CUDAToolkit REQUIRED)
find_package(Thrust)

target_include_directories(${PROJECT_NAME}
                           PRIVATE
			   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link Library
target_link_libraries(${PROJECT_NAME} 
	PRIVATE
	CUDA::cufft
	CUDA::cudart
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
