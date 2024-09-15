#ifndef CMD_GLOBAL_PLACE_H
#define CMD_GLOBAL_PLACE_H

#include "skyline/SkyLine.h"
#include "CmdCore.h"

namespace skyline
{
  class CmdGlobalPlace : public TclCmd
  {
    public:

      CmdGlobalPlace(const char* cmd_name) : TclCmd(cmd_name)
      {
        addOption("-target_density");
        addOption("-target_overflow");
      }

      void execute() override
      {
        double target_ovf = 0.07;
        auto itr_ovf = optArgs_.find("-target_overflow");
        if(itr_ovf != optArgs_.end())
          target_ovf = itr_ovf->second;

        double target_den = 1.0;
        auto itr_den = optArgs_.find("-target_density");
        if(itr_den != optArgs_.end())
          target_den = itr_den->second;

				auto sky = skyline::SkyLine::getStaticPtr();
				sky->runGlobalPlace(target_ovf, target_den);
      }
  };
}

#endif
