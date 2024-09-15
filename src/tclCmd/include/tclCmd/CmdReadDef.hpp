#ifndef CMD_READ_DEF_H
#define CMD_READ_DEF_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdReadDef : public TclCmd
  {
    public:

      CmdReadDef(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
				auto sky = skyline::SkyLine::getStaticPtr();
				sky->readDef(arg_.c_str());
      }
  };
}

#endif
