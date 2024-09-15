#include <cstdlib>
#include <tcl.h>

#include "tclCmd/CmdCore.h"

#include "tclCmd/CmdReadLef.hpp"
#include "tclCmd/CmdReadDef.hpp"
#include "tclCmd/CmdReadVerilog.hpp"
#include "tclCmd/CmdReadBookshelf.hpp"
#include "tclCmd/CmdSetTopModule.hpp"
#include "tclCmd/CmdWriteBookshelf.hpp"
#include "tclCmd/CmdWriteDef.hpp"
#include "tclCmd/CmdGlobalPlace.hpp"
#include "tclCmd/CmdDisplay.hpp"

namespace skyline
{

// This is a callback function when the tcl interpreter meets a command.
int cmdCbk(ClientData clientData, Tcl_Interp* interp, int objc, struct Tcl_Obj* const* objv)
{
  const char* cmd_name = Tcl_GetString(objv[0]);
  skyline::TclCmd* cmd_ptr = TclCmdList::getCmdByName(cmd_name);

  if(!cmd_ptr)
	{
	  printf("Unknown Command : %s\n", cmd_name);
		return TCL_ERROR;
	}

  bool isOptArg = false;
  std::string curOpt = "";

  for(int cnt = 1; cnt < objc; ++cnt) 
  {
    struct Tcl_Obj* obj = objv[cnt];
    const char* obj_str = Tcl_GetString(obj);

    if(obj_str[0] == '-')
    {
      if(isOptArg == true)
      {
        printf("Invalid use of command %s...\n", cmd_name);
        return TCL_ERROR;
      }
      else
      {
        isOptArg = true;
        curOpt = std::string(obj_str);
      }
    }
    else
    {
      if(isOptArg)
      {
        bool status = cmd_ptr->giveOptArg(curOpt, atof(obj_str));
				if(!status)
				{
					printf("Unknown Option : %s\n", curOpt.c_str());
					return TCL_ERROR;
				}
        isOptArg = false;
      }
      else
      {
        std::string obj_cpp_str = std::string(obj_str);
        cmd_ptr->giveArg(obj_cpp_str);
      }
    }
  }

  cmd_ptr->execute();
  return TCL_OK;
}

// Static Class Implementation
std::unordered_map<std::string, std::unique_ptr<TclCmd>> TclCmdList::name2Cmd_;

void
TclCmdList::addTclCmd(Tcl_Interp* interp, std::unique_ptr<TclCmd> newCmd)
{
  Tcl_CreateObjCommand(interp, newCmd->name().c_str(), cmdCbk, nullptr, nullptr);
  // We don't use Client Data and Delete Callback function
  name2Cmd_[newCmd->name()] = std::move(newCmd);
}

TclCmd*
TclCmdList::getCmdByName(const char* name)
{
  std::string nameStr = std::string(name);
  auto itr = name2Cmd_.find(nameStr);

  if(itr != name2Cmd_.end())
    return itr->second.get(); // return raw pointer
  else
    return nullptr;
}

void
CmdCore::initTclCmds(Tcl_Interp* interp)
{
  // DB-related API
  TclCmdList::addTclCmd(interp, std::make_unique<CmdReadLef>("read_lef"));
  TclCmdList::addTclCmd(interp, std::make_unique<CmdReadDef>("read_def"));
  TclCmdList::addTclCmd(interp, std::make_unique<CmdReadVerilog>("read_verilog"));
  TclCmdList::addTclCmd(interp, std::make_unique<CmdReadBookshelf>("read_bookshelf"));
  TclCmdList::addTclCmd(interp, std::make_unique<CmdSetTopModule>("set_top_module"));

  TclCmdList::addTclCmd(interp, std::make_unique<CmdWriteDef>("write_def"));
  TclCmdList::addTclCmd(interp, std::make_unique<CmdWriteBookshelf>("write_bookshelf"));

	// GUI-related API
  TclCmdList::addTclCmd(interp, std::make_unique<CmdDisplay>("display"));

	// Engine-related API
  TclCmdList::addTclCmd(interp, std::make_unique<CmdGlobalPlace>("global_place"));
}

} // namespace skyline
