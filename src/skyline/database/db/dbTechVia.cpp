#include <iostream>
#include "dbTechVia.h"

namespace db
{

dbTechVia::dbTechVia() 
  : isDefault_  (false),
    hasViaRule_ (false),
    topLayer_   (nullptr),
    cutLayer_   (nullptr),
    botLayer_   (nullptr)
{
}

void
dbTechVia::setBoundary()
{
  bool cutLayerFound = false;

  for(const auto rect : rects_)
  {
    if(rect->layer() == cutLayer_)
    {
      int lx = rect->lx();
      int ly = rect->ly();
      int ux = rect->ux();
      int uy = rect->uy();
      shape_.push_back( std::pair<int, int>(lx, ly) );
      shape_.push_back( std::pair<int, int>(ux, uy) );
      cutLayerFound = true;
      break;
    }
  }

  for(const auto poly : polygons_)
  {
    if(poly->layer() == cutLayer_)
    {
      if(cutLayerFound == false)
      {
        shape_ = poly->getShape();
        cutLayerFound = true;
        break;
      }
      else
      {
        std::cout << "There are both RECT and POLYGON";
        std::cout << " in the CUT layer of the via ";
        std::cout << name_ << std::endl;
        exit(1);
      }
    }
  }
}

void
dbTechVia::print() const
{
  std::cout << std::endl;
  std::cout << "VIA : " << name_    << std::endl;
  std::cout << "TOP LAYER : " << topLayer_->name() << std::endl;
  std::cout << "CUT LAYER : " << cutLayer_->name() << std::endl;
  std::cout << "BOT LAYER : " << botLayer_->name() << std::endl;
  std::cout << "SHAPE" << std::endl;
  for(auto& [posX, posY] : shape_)
    std::cout << posX << " " << posY << std::endl;
  std::cout << std::endl;
}

}
