#include "skygui/SkyGui.h"
#include "MainWindow.h"

#include <QApplication>

extern int    cmd_argc;
extern char** cmd_argv; 

namespace gui
{

SkyGui::SkyGui(std::shared_ptr<dbDatabase> db)
  : db_ (db)
{
}

SkyGui::~SkyGui()
{
}

void
SkyGui::display()
{
  QApplication app(cmd_argc, cmd_argv);
  MainWindow window;
  window.setDatabase(db_);
  window.init();
  window.show();
  int exit_code = app.exec();
  //exit(exit_code);
}

}
