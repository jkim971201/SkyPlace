#include <iostream>

#include "dbMacro.h"

namespace db
{

dbMacro::dbMacro()
  : name_   (""),
    site_   (nullptr),
    sizeX_  (0),
    sizeY_  (0),
    origX_  (0),
    origY_  (0),
    symX_   (false),
    symY_   (false),
    symR90_ (false)
{
  mterms_.clear();
  mtermMap_.clear();
  macroClass_ = MacroClass::CORE;
}

dbMacro::~dbMacro()
{
  mterms_.clear();
  mtermMap_.clear();
}

void
dbMacro::addMTerm(dbMTerm* newPin)
{
  mterms_.push_back(newPin);
  mtermMap_[newPin->name()] = newPin;
}

dbMTerm*
dbMacro::getMTermByName(const std::string& pinName)
{
  auto itr = mtermMap_.find(pinName);
  
  if(itr == mtermMap_.end())
  {
    std::cout << "Cannot find Pin " << pinName;
    std::cout << " in MACRO << " << name_ << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

void
dbMacro::print() const
{
  std::cout << std::endl;
  std::cout << "MACRO        : " << name_         << std::endl;
  std::cout << "CLASS        : " << macroClass_   << std::endl;
  std::cout << "SIZE X       : " << sizeX_        << std::endl;
  std::cout << "SIZE Y       : " << sizeY_        << std::endl;
  std::cout << "ORIGIN X     : " << origX_        << std::endl;
  std::cout << "ORIGIN Y     : " << origY_        << std::endl;
  std::cout << "SYMMETRY X   : " << symX_         << std::endl;
  std::cout << "SYMMETRY Y   : " << symY_         << std::endl;
  std::cout << "SYMMETRY R90 : " << symR90_       << std::endl;
  std::cout << std::endl;

  for(const auto pin : mterms_)
    pin->print();
}

}
