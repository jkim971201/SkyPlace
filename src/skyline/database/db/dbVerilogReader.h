#ifndef DB_VERILOG_READER_H
#define DB_VERILOG_READER_H

#include <vector>
#include <string>
#include <memory>

#include "verilog_driver.hpp"

#include "dbUtil.h"
#include "dbTech.h"
#include "dbDesign.h"

namespace db
{

class Module;
class ModuleInst;

typedef std::shared_ptr<Module>     ModulePtr;
typedef std::shared_ptr<ModuleInst> ModuleInstPtr;

class ModuleInst
{
  public:

    ModuleInst() {}
    ModuleInst(const std::string& name, ModulePtr mod, ModulePtr parent)
      : name_      (name),
        hierName_  (""),
        module_    (mod),
        parent_    (parent)
    {}
 
    const std::string& name()     const { return name_;     }
    const std::string& hierName() const { return hierName_; }

    bool isTopModule() const { return parent_ == nullptr; }

    ModulePtr getModule() { return module_; }
    ModulePtr getParent() { return parent_; }
 
    void setName     (const std::string& name) { name_     = name; }
    void setHierName (const std::string& name) { hierName_ = name; }

  private:

    // HierName is hierarchical name (e.g. A/B/C)
    std::string name_;
    std::string hierName_;
    ModulePtr module_;
    ModulePtr parent_;
};

class Module
{
  public:

    Module() {}
    Module(const std::string& name) : name_ (name) 
    {
      ports_.clear();
      insts_.clear();
			str2moduleInst_.clear();
    }

    const std::string& name() const { return name_; }

    void addPort(verilog::Port&     port) { ports_.push_back(port); }
    void addInst(verilog::Instance& inst) { insts_.push_back(inst); }

    const std::vector<verilog::Port>&     ports() const { return ports_; }
    const std::vector<verilog::Instance>& insts() const { return insts_; }

    void addChild(ModuleInstPtr modInst) 
		{
			const std::string instName = modInst->name();
			if(duplicateCheck(str2moduleInst_, instName) == true)
			{
				printf("Instance %s already exists in module %s\n",
				        instName.c_str(), name_.c_str());
				exit(1);
			}
	    children_.push_back(modInst); 
	    str2moduleInst_[modInst->name()] = modInst;
		}

    std::vector<ModuleInstPtr>& getChilren() { return children_; }

    bool isLeafModule() const { return children_.empty(); }

  private:

    std::string name_;
    std::vector<ModuleInstPtr> children_;
    std::vector<verilog::Port> ports_;
    std::vector<verilog::Instance> insts_;
    std::unordered_map<std::string, ModuleInstPtr> str2moduleInst_;
};

class dbVerilogReader : public verilog::ParserVerilogInterface
{
  public:
  
    dbVerilogReader(std::shared_ptr<dbTech>   tech,
                    std::shared_ptr<dbDesign> design);

    void readFile(const std::string& filename);

    void setTopModuleName(const std::string& name) { topModuleName_ = name; }

    const std::vector<ModulePtr> getModules() const { return modules_; }
    const ModuleInstPtr getTopModuleInst() const { return topModuleInst_; }

    // Virtual methods (from verilog::ParserVerilogInterface)
    virtual ~dbVerilogReader() {}

    // Module Callback
    virtual void add_module(std::string&& name) override;

    // Port Callback (IO Pin)
    virtual void add_port(verilog::Port&& port) override;

    // Net Callback
    virtual void add_net(verilog::Net&& net) override;

    // Assignment Callback
    virtual void add_assignment(verilog::Assignment&& ast) override; 

    // Instance Callback
    virtual void add_instance(verilog::Instance&& inst) override; 
    
  private:

    std::string topModuleName_;

    ModulePtr     makeNewModule     (const std::string& name);
    ModuleInstPtr makeNewModuleInst (const std::string& name, ModulePtr mod, ModulePtr parent);

    std::shared_ptr<dbTech>   tech_;
    std::shared_ptr<dbDesign> design_;

    ModulePtr     curModule_;
    ModuleInstPtr topModuleInst_;

    std::vector<verilog::Net> nets_;
    std::vector<verilog::Assignment> assignments_;

    std::vector<ModulePtr> modules_;
    std::vector<ModuleInstPtr> moduleInsts_;
    std::unordered_map<std::string, ModulePtr> str2module_;
};

}

#endif
