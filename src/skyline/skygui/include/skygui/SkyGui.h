#ifndef SKY_GUI_H
#define SKY_GUI_H

#include <memory>

#include "db/dbDatabase.h"

using namespace db;

namespace gui
{

class SkyGui
{
  public:

    SkyGui(std::shared_ptr<dbDatabase> db);
    ~SkyGui();

    void linkDatabase(std::shared_ptr<dbDatabase> db);
    void display();

  private:

    std::shared_ptr<dbDatabase> db_;
};

}

#endif
