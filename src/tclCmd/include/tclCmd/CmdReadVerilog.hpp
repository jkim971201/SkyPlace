#ifndef CMD_READ_VERILOG_H
#define CMD_READ_VERILOG_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdReadVerilog : public TclCmd
  {
    public:

      CmdReadVerilog(const char* cmd_name) : TclCmd(cmd_name)
      {
      }

      void execute() override
      {
				auto sky = skyline::SkyLine::getStaticPtr();
				sky->readVerilog(arg_.c_str());
      }
  };
}

#endif
