#ifndef CMD_READ_BOOKSHELF_H
#define CMD_READ_BOOKSHELF_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdReadBookshelf : public TclCmd
  {
    public:

      CmdReadBookshelf(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
				auto sky = skyline::SkyLine::getStaticPtr();
				sky->readBookShelf(arg_.c_str());
      }
  };
}

#endif
