#include <iostream>
#include <cassert>

#include "dbInst.h"
#include "dbMacro.h"
#include "dbITerm.h"

namespace db
{

dbInst::dbInst()
  : macro_ (nullptr),
    name_  (""),
    lx_    (0),
    ly_    (0),
    dx_    (0),
    dy_    (0),
    haloT_ (0),
    haloB_ (0),
    haloL_ (0),
    haloR_ (0)
{
  orient_ = Orient::N;
  source_ = Source::NETLIST;
  status_ = Status::UNPLACED;
  iterms_.clear();
  itermMap_.clear();
}

bool
dbInst::isFixed() const 
{
  return status_ == Status::FIXED || status_ == Status::COVER;
}

bool
dbInst::isStdCell() const 
{
  return macro_->macroClass() == MacroClass::CORE          
     ||  macro_->macroClass() == MacroClass::CORE_FEEDTHRU 
     ||  macro_->macroClass() == MacroClass::CORE_TIELOW   
     ||  macro_->macroClass() == MacroClass::CORE_TIEHIGH  
     ||  macro_->macroClass() == MacroClass::CORE_SPACER   
     ||  macro_->macroClass() == MacroClass::CORE_WELLTAP  
     ||  macro_->macroClass() == MacroClass::CORE_ANTENNACELL;
}

bool
dbInst::isMacro() const 
{
  return macro_->macroClass() == MacroClass::BLOCK;
}

bool
dbInst::isDummy() const 
{
  return source_ == Source::DIST;
}

void 
dbInst::setLocation(int placementX, int placementY)
{
  switch(orient_)
  {
    case Orient::N :
    {
      dx_ = macro_->sizeX();
      dy_ = macro_->sizeY();
      lx_ = placementX;
      ly_ = placementY;
      break;
    }
    case Orient::S :
    {
      dx_ = macro_->sizeX();
      dy_ = macro_->sizeY();
      lx_ = placementX;
      ly_ = placementY;

      //lx_ = placementX - dx_;
      //ly_ = placementY - dy_;
      break;
    }
    case Orient::W :
    {
      dx_ = macro_->sizeY();
      dy_ = macro_->sizeX();
      lx_ = placementX;
      ly_ = placementY;

      //lx_ = placementX - dx_;
      //ly_ = placementY;
      break;
    }
    case Orient::E :
    {
      dx_ = macro_->sizeY();
      dy_ = macro_->sizeX();
      lx_ = placementX;
      ly_ = placementY;

      //lx_ = placementX;
      //ly_ = placementY - dy_;
      break;
    }
    case Orient::FN :
    {
      dx_ = macro_->sizeX();
      dy_ = macro_->sizeY();
      lx_ = placementX;
      ly_ = placementY;

      //lx_ = placementX - dx_;
      //ly_ = placementY;
      break;
    }
    case Orient::FS :
    {
      dx_ = macro_->sizeX();
      dy_ = macro_->sizeY();
      lx_ = placementX;
      ly_ = placementY;

      //lx_ = placementX;
      //ly_ = placementY - dy_;
      break;
    }
    case Orient::FW :
    {
      dx_ = macro_->sizeY();
      dy_ = macro_->sizeX();

      lx_ = placementX;
      ly_ = placementY;
      break;
    }
    case Orient::FE :
    {
      dx_ = macro_->sizeY();
      dy_ = macro_->sizeX();
      lx_ = placementX;
      ly_ = placementY;

      //lx_ = placementX - dx_;
      //ly_ = placementY - dy_;
      break;
    }
  }
}

void
dbInst::addITerm(dbITerm* iterm)
{
  iterms_.push_back(iterm);

  std::string nameMTerm = iterm->getMTerm()->name();

  if(itermMap_.find(nameMTerm) == itermMap_.end())
    itermMap_[nameMTerm] = iterm;
  else
  {
    std::cout << "Port " << nameMTerm << " already exist in ";
    std::cout << name_ << std::endl;
    assert(0);
  }
}

dbITerm* 
dbInst::getITermByMTermName(const std::string& name)
{
  auto itr = itermMap_.find(name);

  if(itr == itermMap_.end())
	{
		std::cout << "Wrong Port Name : " << name << std::endl;
		assert(0);
	}
  else
		return itr->second;
}

void
dbInst::print() const
{
  std::cout << std::endl;
  std::cout << "MACRO : " << macro_->name() << std::endl;
  std::cout << "NAME  : " << name_ << std::endl;
  std::cout << "LX LY : " << lx() << " " << ly() << std::endl;
  std::cout << "UX UY : " << ux() << " " << uy() << std::endl;
  std::cout << std::endl;
}

}
