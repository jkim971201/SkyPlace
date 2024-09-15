add_library(tcl_cmd)

find_package(TCL)

target_sources(tcl_cmd
  PRIVATE
	src/CmdCore.cpp
)

target_include_directories(tcl_cmd
  PUBLIC
	include

	PRIVATE
)

target_link_libraries(tcl_cmd
	PRIVATE
  skyline
	${TCL_LIBRARY}
)
