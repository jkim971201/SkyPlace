find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

set(CMAKE_C_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)

BISON_TARGET(verilog_parser
             src/verilog_parser.yy
             ${CMAKE_CURRENT_BINARY_DIR}/verilog_parser.tab.cc)

FLEX_TARGET(verilog_lexer
            src/verilog_lexer.l
            ${CMAKE_CURRENT_BINARY_DIR}/verilog_lexer.yy.cc)

ADD_FLEX_BISON_DEPENDENCY(verilog_lexer verilog_parser)

add_library(verilog
  ${FLEX_verilog_lexer_OUTPUTS}
  ${BISON_verilog_parser_OUTPUTS}
)

# include directory settings
target_include_directories(verilog
  PUBLIC
    include
    ${CMAKE_CURRENT_BINARY_DIR}
)

#MESSAGE( STATUS "CURRENT BINARY PATH: ${CMAKE_CURRENT_BINARY_DIR}")
#get_target_property(include_path_list verilog INCLUDE_DIRECTORIES)
#foreach( dir ${include_path_list})
#MESSAGE( STATUS "include path: ${dir}")
#endforeach()
