#include <iostream>
#include <string>
#include <tcl.h>

#include "tclCmd/CmdCore.h"
#include "skyline/SkyLine.h"

bool exit_mode = true;

int    cmd_argc;
char** cmd_argv;

void sourceTclFile(std::string& filename, Tcl_Interp *interp)
{
  std::string cmd = "source " + filename;
  int code = Tcl_Eval(interp, cmd.c_str());

  const char* result = Tcl_GetStringResult(interp);
  if(result[0] != '\0')
    std::cout << result << std::endl;
  if(exit_mode)
    exit(0);
}

int customTclInit(char** argv, Tcl_Interp* interp)
{
  interp = Tcl_CreateInterp();

  if(Tcl_Init(interp) == TCL_ERROR)
  {
    std::cout << "ERROR - Cannot create Tcl interpreter" << std::endl;
    return TCL_ERROR;
  }
  else
  {
		skyline::CmdCore::initTclCmds(interp);

    std::string filename = std::string(argv[1]);

    sourceTclFile(filename, interp);

    return TCL_OK;
  }
}

int tclInitWrapper(Tcl_Interp* interp)
{
  return customTclInit(cmd_argv, interp);
}

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "Please give input file" << std::endl;
    exit(0);
  }

	cmd_argc = argc;
  cmd_argv = argv;

  Tcl_Main(1, argv, tclInitWrapper);

  return 0;
}
