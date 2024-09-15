#ifndef CMD_READ_LEF_H
#define CMD_READ_LEF_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdReadLef : public TclCmd
  {
    public:

      CmdReadLef(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
				auto sky = skyline::SkyLine::getStaticPtr();
				sky->readLef(arg_.c_str());
      }
  };
}

#endif
