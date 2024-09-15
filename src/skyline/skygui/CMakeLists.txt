set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Widgets)

set(gui_sources 
    src/SkyGui.cpp
    src/MainWindow.cpp
    src/LayoutView.cpp
    src/LayoutScene.cpp
    src/gui_item/GuiDie.cpp
    src/gui_item/GuiRow.cpp
    src/gui_item/GuiInst.cpp
    src/gui_item/GuiIO.cpp
    src/gui_item/GuiPin.cpp
    src/gui_item/GuiNet.cpp
    src/gui_item/GuiRect.cpp
    src/gui_item/GuiBlockage.cpp
    src/gui_item/GuiTrackGrid.cpp
    icons/resource.qrc
)

set(gui_headers
    src/MainWindow.h
    src/LayoutView.h
    src/LayoutScene.h
    src/gui_item/GuiDie.h
    src/gui_item/GuiRow.h
    src/gui_item/GuiInst.h
    src/gui_item/GuiPin.h
    src/gui_item/GuiRect.h
    src/gui_item/GuiNet.h
    src/gui_item/GuiItem.h
    src/gui_item/GuiBlockage.h
    src/gui_item/GuiTrackGrid.h
)

# QMake re-compile the MOC files 
# when headers (with Q_OBJECT) are added in library list)

add_library(gui 
  ${gui_sources} 
  ${gui_headers}
)

target_link_libraries(gui
  PRIVATE
  db
  Qt${QT_VERSION_MAJOR}::Widgets
)

target_include_directories(gui
  PUBLIC
  include

  PRIVATE
  src
)
