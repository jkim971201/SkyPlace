#include <iostream>
#include <limits>

#include "dbMTerm.h"

namespace db
{

dbMTerm::dbMTerm()
  : macro_   (nullptr),
    name_    (""),
    lx_      (0),
    ly_      (0),
    ux_      (0),
    uy_      (0)
{
}

void
dbMTerm::setBoundary()
{
  if(ports_.empty())
    return;

  lx_ = std::numeric_limits<int>::max();
  ly_ = std::numeric_limits<int>::max();
  ux_ = std::numeric_limits<int>::min();
  uy_ = std::numeric_limits<int>::min();

  for(auto& port : ports_)
  {
    for(auto& [newX, newY] : port->getShape())
    {
      if(newX < lx_) lx_ = newX;
      if(newX < ly_) ly_ = newY;
      if(newY > ux_) ux_ = newX;
      if(newY > uy_) uy_ = newY;
    }
  }
}

void
dbMTerm::print() const
{
  std::cout << std::endl;
  std::cout << "PIN       : " << name_  << std::endl;
  std::cout << "MACRO     : " << macro_->name() << std::endl;
  std::cout << "DIRECTION : " << pinDir_   << std::endl;
  std::cout << "USAGE     : " << pinUsage_ << std::endl;
  std::cout << "SHAPE     : " << pinShape_ << std::endl;
  for(const auto& p : ports_)
    std::cout << "LAYER: " << p->layer()->name() << " ";
  std::cout << std::endl;
}

}
