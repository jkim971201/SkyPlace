#ifndef CMD_DISPLAY_H
#define CMD_DISPLAY_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdDisplay : public TclCmd
  {
    public:

      CmdDisplay(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
				auto sky = skyline::SkyLine::getStaticPtr();
				sky->display();
      }
  };
}

#endif
