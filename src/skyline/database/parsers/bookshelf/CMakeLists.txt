add_library(bookshelf
	BookShelfParser.cpp
)

# include directory settings
target_include_directories(bookshelf
  PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)
