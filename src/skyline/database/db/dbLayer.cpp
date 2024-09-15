#include <iostream>

#include "dbLayer.h"

namespace db
{

dbLayer::dbLayer()
  : name_     (""),
    xPitch_   ( 0),
    yPitch_   ( 0),
    xOffset_  ( 0),
    yOffset_  ( 0),
    width_    ( 0),
    spacing_  ( 0),
    area_     ( 0)
{
}

void
dbLayer::print() const
{
  std::cout << std::endl;
  std::cout << "LAYER     : " << name_    << std::endl;
  std::cout << "TYPE      : " << type_    << std::endl;
  std::cout << "DIRECTION : " << dir_     << std::endl;
  std::cout << "X PITCH   : " << xPitch_  << std::endl;
  std::cout << "Y PITCH   : " << yPitch_  << std::endl;
  std::cout << "X OFFSET  : " << xOffset_ << std::endl;
  std::cout << "Y OFFSET  : " << yOffset_ << std::endl;
  std::cout << "WIDTH     : " << width_   << std::endl;
  std::cout << "SPACING   : " << spacing_ << std::endl;
  std::cout << "AREA      : " << area_    << std::endl;
  std::cout << std::endl;
}

}
