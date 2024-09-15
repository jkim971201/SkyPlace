#include <iostream>

#include "dbSite.h"

namespace db
{

dbSite::dbSite()
  : name_   (""),
    sizeX_  ( 0),
    sizeY_  ( 0),
    symX_   (false),
    symY_   (false),
    symR90_ (false)
{
}

void
dbSite::print() const
{
  std::cout << std::endl;
  std::cout << "SITE  : " << name_      << std::endl;
  std::cout << "CLASS : " << siteClass_ << std::endl;
  std::cout << sizeX_ << " BY " << sizeY_ << std::endl;
  std::cout << "SYMMETRY X   : " << symX_   << std::endl;
  std::cout << "SYMMETRY Y   : " << symY_   << std::endl;
  std::cout << "SYMMETRY R90 : " << symR90_ << std::endl;
  std::cout << std::endl;
}

}
