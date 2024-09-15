#include "dbITerm.h"
#include "dbMTerm.h"
#include "dbInst.h"
#include "dbTypes.h"
#include "dbRect.h"

#include <iostream>

namespace db
{

dbITerm::dbITerm()
  : name_  (""),
    net_   (nullptr),
    inst_  (nullptr),
    mterm_ (nullptr)
{
}

dbITerm::dbITerm(const std::string& name, dbInst* inst, dbMTerm* mterm)
  : name_  (name),
    net_   (nullptr),
    inst_  (inst),
    mterm_ (mterm)
{
}

const dbRect
dbITerm::getRect() const
{
  auto orient = inst_->orient();

  int lx = 0;
  int ly = 0;
  int dx = mterm_->dx();
  int dy = mterm_->dy();

  switch(orient)
  {
		case Orient::N :
    {
			lx = inst_->lx() + mterm_->lx();
			ly = inst_->ly() + mterm_->ly();
			break;
    }
		case Orient::S :
    {
			lx = inst_->ux() - mterm_->lx();
			ly = inst_->uy() - mterm_->ly();
			break;
    }
		case Orient::W :
    {
			lx = inst_->ux() - mterm_->ly() - dy;
			ly = inst_->ly() + mterm_->ly();
			break;
    }
		case Orient::E :
    {
			lx = inst_->lx() + mterm_->ly();
			ly = inst_->uy() - mterm_->lx() - dx;
			break;
    }
		case Orient::FN :
    {
			lx = inst_->ux() - mterm_->lx() - dx;
			ly = inst_->ly() + mterm_->ly();
			break;
    }
		case Orient::FS :
    {
			lx = inst_->lx() + mterm_->lx();
			ly = inst_->uy() - mterm_->uy() - dy;
			break;
    }
		case Orient::FW :
    {
			lx = inst_->lx() + mterm_->ly();
			ly = inst_->ly() + mterm_->lx();
			break;
    }
		case Orient::FE :
    {
			lx = inst_->ux() - mterm_->lx() - dx;
			ly = inst_->uy() - mterm_->ly() - dy;
			break;
    }
  }

	return dbRect(lx, ly, lx + dx, ly + dy, nullptr);
}

bool 
dbITerm::isSignal() const
{
  return mterm_->usage() == PinUsage::SIGNAL;
}

void
dbITerm::print() const
{
  std::cout << "ITerm Name : " << name_ << std::endl;
  //if(net_ != nullptr)
  //  std::cout << "Connected to net: " << net_->name() << std::endl;
  //std::cout << std::endl;
}

}
