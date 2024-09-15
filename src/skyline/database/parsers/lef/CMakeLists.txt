# BISON is required to compile LEF/LEF Parsers.
find_package(BISON REQUIRED)

MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

bison_target(lef_parser
             ${CMAKE_CURRENT_SOURCE_DIR}/lef/lef.y
             ${CMAKE_CURRENT_BINARY_DIR}/lef_parser.cpp
             COMPILE_FLAGS "--name-prefix=lefyy"
)

############################################################
# Create a library for LEF
############################################################

MESSAGE(STATUS "BISON_lef_parser_OUTPUT_SOURCE: ${BISON_lef_parser_OUTPUT_SOURCE}")

add_library(lef
  lef/crypt.cpp
  lef/lefiArray.cpp
  lef/lefiCrossTalk.cpp
  lef/lefiDebug.cpp
  lef/lefiEncryptInt.cpp
  lef/lefiLayer.cpp
  lef/lefiMacro.cpp
  lef/lefiMisc.cpp
  lef/lefiNonDefault.cpp
  lef/lefiProp.cpp
  lef/lefiPropType.cpp
  lef/lefiTBExt.cpp
  lef/lefiUnits.cpp
  lef/lefiVia.cpp
  lef/lefiViaRule.cpp
  lef/lef_keywords.cpp
  lef/lefrCallbacks.cpp
  lef/lefrData.cpp
  lef/lefrReader.cpp
  lef/lefrSettings.cpp
  lef/lefwWriterCalls.cpp
  lef/lefwWriter.cpp
  ${BISON_lef_parser_OUTPUT_SOURCE}
)


set_target_properties(lef 
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# include directory settings
target_include_directories(lef
  PUBLIC
	  lef
	  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR} # for lef_parser.hpp
)

target_compile_options(lef
  PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-class-memaccess>
)
