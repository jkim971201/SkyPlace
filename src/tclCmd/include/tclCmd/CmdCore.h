#ifndef CMD_CORE_H
#define CMD_CORE_H

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>

extern "C" {
struct Tcl_Interp;
}

namespace skyline
{

class TclCmd
{
 public:

  TclCmd(const char* cmd_name)
  {
    name_ = std::string(cmd_name);
    arg_ = "";
    opts_.clear();
    optArgs_.clear();
  }

  ~TclCmd() {}

  const std::string& name() { return name_; }

  void giveArg(std::string& arg) { arg_ = arg; }

  bool giveOptArg(std::string& opt_name, double number)
  {
    if(std::find(opts_.begin(), opts_.end(), opt_name) != opts_.end())
    {
      optArgs_[opt_name] = number;
      return true;
    }

    return false;
  }

  virtual void execute() = 0; // Pure Virtual Function

 protected:

  void addOption(std::string option_name)
  {
    opts_.push_back(option_name);
    // optArgs_[option_name] = 0.0;
  }

  std::string name_;
  std::string arg_;                                 // Argument of this command 
  std::vector<std::string> opts_;                   // Options  of this command
  std::unordered_map<std::string, double> optArgs_; // Argument of Options
};

class TclCmdList
{
  public:
    static void addTclCmd(Tcl_Interp* interp, std::unique_ptr<TclCmd> newCmd);
    static TclCmd* getCmdByName(const char* name);

  private:
    static std::unordered_map<std::string, std::unique_ptr<TclCmd>> name2Cmd_;
};

class CmdCore
{
  public:
    // Tcl Interface
    static void initTclCmds(Tcl_Interp* interp);
};

}

#endif
