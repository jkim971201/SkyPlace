#ifndef DB_NDR_H
#define DB_NDR_H

#include <unordered_map>
#include <string>

#include "dbLayer.h"

namespace db
{

class dbLayer;
class dbViaMaster;

struct dbLayerRule
{
  dbLayer* layer = nullptr;
  int width = 0;
  int spacing = 0;
};

class dbNonDefaultRule
{
  public:

    dbNonDefaultRule() : isHardSpacing_(false) {}

    // Getters
    const std::string& name() { return name_; }
    bool isHardSpacing() const { return isHardSpacing_; }
    dbLayerRule* getLayerRuleByName(const std::string& layerName) const
    {
      auto itr = str2rules_.find(layerName);
  
      if(itr == str2rules_.end())  
        return nullptr;
      else
        return itr->second;
    }

    // Setters
    void setName(const std::string& name) { name_ = name; }
    void setHardSpacing() { isHardSpacing_ = true; }

    void addVia(dbViaMaster* via) { vias_.push_back(via); }
    void addLayerRule(dbLayerRule* rule) 
    { 
      str2rules_[rule->layer->name()] = rule;
      rules_.push_back(rule); 
    }

  private:

    bool isHardSpacing_;
    std::string name_;
    std::vector<dbLayerRule*> rules_;
    std::vector<dbViaMaster*> vias_;
    std::unordered_map<std::string, dbLayerRule*> str2rules_;
};

}

#endif
