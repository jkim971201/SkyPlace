#ifndef CMD_SET_TOP_MODULE_H
#define CMD_SET_TOP_MODULE_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdSetTopModule : public TclCmd
  {
    public:

      CmdSetTopModule(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
				auto sky = skyline::SkyLine::getStaticPtr();
				sky->setTopModuleName(arg_.c_str());
      }
  };
}

#endif
