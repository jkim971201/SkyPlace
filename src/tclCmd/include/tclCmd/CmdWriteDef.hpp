#ifndef CMD_WRITE_DEF_H
#define CMD_WRITE_DEF_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdWriteDef : public TclCmd
  {
    public:

      CmdWriteDef(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
        auto sky = skyline::SkyLine::getStaticPtr();
        sky->writeDef(arg_.c_str());
      }
  };
}

#endif
