#ifndef CMD_WRITE_BOOKSHELF_H
#define CMD_WRITE_BOOKSHELF_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdWriteBookshelf : public TclCmd
  {
    public:

      CmdWriteBookshelf(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
        auto sky = skyline::SkyLine::getStaticPtr();
        sky->writeBookShelf(arg_.c_str());
      }
  };
}

#endif
