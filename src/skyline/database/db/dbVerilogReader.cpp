#include <cstdio>
#include <algorithm>

#include "dbVerilogReader.h"

namespace db
{

dbVerilogReader::dbVerilogReader(std::shared_ptr<dbTech>   tech,
                                 std::shared_ptr<dbDesign> design)
  : topModuleName_  (""),
    tech_           (   tech),
    design_         ( design),
    curModule_      (nullptr),
    topModuleInst_  (nullptr)
{
  nets_.clear();
  assignments_.clear();
  modules_.clear();
  moduleInsts_.clear();
  str2module_.clear();
}

void
dbVerilogReader::readFile(const std::string& filename)
{
  size_t dot = filename.find_last_of('.');
  std::string filetype = filename.substr(dot + 1);

  if(filetype != "v")
  {
    printf("Please give .v file!\n");
    exit(1);
  }

  if(topModuleName_ == "")
  {
    if(design_->name() != "")
    {
      // top module name is not given,
      // then use the design name in the .def file
      topModuleName_ = design_->name();
    }
    else
    {
      // if def is not read yet, and even top module name is not set,
      // then it must be an error.
      printf("You should give top module name first!\n");
      exit(1);
    }
  }

  // read function of ParserVerilogInterface will do
  // file existence check
  bool is_success = this->read(filename);

  if(!is_success)
  {
    printf("Fail to open %s\n", filename.c_str());
    exit(1);
  }

  if(topModuleInst_ == nullptr)
  {
    printf("Top module %s does not exist in the verilog!\n", topModuleName_.c_str());
    exit(1);
  }

  // Set hierarchical name
  // recusrsive use of lambda cannot use "auto"
  std::function<void(ModuleInstPtr, const std::string&)> resetHierNameRecur 
    = [&] (ModuleInstPtr modInst, const std::string& hierName)
  {
    ModulePtr masterMod = modInst->getModule();

    const std::string hierNameThisLevel 
      = (modInst == topModuleInst_) ? hierName : hierName + "/" + modInst->name();

    modInst->setHierName(hierNameThisLevel);
    for(auto child : masterMod->getChilren())
      resetHierNameRecur(child, hierNameThisLevel);
  };

  resetHierNameRecur(topModuleInst_, topModuleName_);

//  for(auto mod : modules_)
//  {
//    printf("Module Name : %s\n", mod->name().c_str());
//    for(auto child : mod->getChilren())
//    {
//      printf("  Children      Name : %s\n", child->name().c_str());
//      printf("  Children Hier Name : %s\n", child->hierName().c_str());
//    }
//  }
}

void 
dbVerilogReader::add_module(std::string&& name)
{
  // std::cout << "Module name = " << name << '\n';

  auto itr = str2module_.find(name);

  if(itr == str2module_.end())
    curModule_ = makeNewModule(name);
  else
    curModule_ = itr->second;

  // NOTE
  // Top Module has no instance itself, so we have to create 
  // moduleInst of Top module here.
  // This makes code dirty, but can't come up with better idea.
  if(name == topModuleName_)
    topModuleInst_ = makeNewModuleInst(topModuleName_, curModule_, nullptr);
}
  
void 
dbVerilogReader::add_port(verilog::Port&& port) 
{
  //std::cout << "Port: " << port << '\n';
  //ports_.push_back(std::move(port));
}  
  
void 
dbVerilogReader::add_net(verilog::Net&& net) 
{
  //std::cout << "Net: " << net << '\n';
  nets_.push_back(std::move(net));
}  
  
void 
dbVerilogReader::add_assignment(verilog::Assignment&& ast) 
{
  // std::cout << "Assignment: " << ast << '\n';
  assignments_.push_back(std::move(ast));
}  
  
void 
dbVerilogReader::add_instance(verilog::Instance&& inst) 
{
  // std::cout << "Instance: " << inst << '\n';
  const std::string modName  = inst.module_name;
  const std::string instName = inst.inst_name;

  dbMacro* techCell = tech_->getMacroByName(modName);

  if(techCell != nullptr) // Tech exists
  {
    curModule_->addInst(inst);
  }
  else // Tech does not exist -> check module
  {
    ModulePtr mod;
    auto itr = str2module_.find(modName);
    if(itr == str2module_.end()) // module name does not exist
    {
      mod = makeNewModule(modName);
    }
    else
    {
      mod = itr->second;
    }

    ModuleInstPtr newModuleInst // Inst Name, Master Module, Parent Module
      = makeNewModuleInst(instName, mod, curModule_);
    curModule_->addChild(newModuleInst);
  }
}

ModulePtr
dbVerilogReader::makeNewModule(const std::string& name)
{
  ModulePtr newModulePtr = std::make_shared<Module>(name);

  // Duplicate Check
  assert(duplicateCheck(str2module_, name) == false);

  str2module_[name] = newModulePtr;
  modules_.push_back(newModulePtr);
  return newModulePtr;
}

ModuleInstPtr
dbVerilogReader::makeNewModuleInst(const std::string& instName, ModulePtr mod, ModulePtr parent)
{
  // Instance Name will be updated to hierarchical name later (end of readVerilog)
  ModuleInstPtr newModuleInstPtr 
    = std::make_shared<ModuleInst>(instName, mod, parent);

  // Duplicate Check will be done in addChild of Module Class
  moduleInsts_.push_back(newModuleInstPtr);
  return newModuleInstPtr;
}

}
