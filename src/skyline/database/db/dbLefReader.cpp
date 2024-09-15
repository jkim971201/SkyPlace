#include <cassert>
#include <cstdio>
#include <string>

#include "dbLefReader.h"

namespace db
{

// static variable initialization
dbMacro* dbLefReader::topMacro_ = nullptr;

dbLefReader::dbLefReader(std::shared_ptr<dbTypes> types, 
                         std::shared_ptr<dbTech> tech)
  : types_          ( types),
    tech_           (  tech),
    parse65nm_      (     0),
    parseLef58Type_ (     0)
{
}

void
dbLefReader::init()
{
  lefrInitSession(0);
  lefrReset();
  
  lefrSetUserData(tech_.get());

  lefrSetUnitsCbk(this->lefUnitsCbk);

  lefrSetBusBitCharsCbk(this->lefBusBitCbk);
  lefrSetDividerCharCbk(this->lefDividerCbk);

  lefrSetLayerCbk(this->lefLayerCbk);
  lefrSetSiteCbk(this->lefSiteCbk);
  lefrSetMacroBeginCbk(this->lefMacroBeginCbk);
  lefrSetMacroCbk(this->lefMacroCbk);
  lefrSetPinCbk(this->lefMacroPinCbk);
  lefrSetObstructionCbk(this->lefMacroObsCbk);
  lefrSetMacroEndCbk(this->lefEndCbk);
	lefrSetViaCbk(this->lefViaCbk);
}

void
dbLefReader::parseLef(const std::string& filename)
{
  init();

  size_t dot = filename.find_last_of('.');
  std::string filetype = filename.substr(dot + 1);

  if(filetype != "lef")
  {
    printf("Please give .lef file!\n");
    exit(1);
  }

  FILE* file = fopen(filename.c_str(), "r");

  if(file == nullptr)
  {
    printf("Cannot open %s\n", filename.c_str());
    exit(1);
  }

  int res = lefrRead(file, filename.c_str(), (void*) tech_.get());
  fclose(file);

  lefrClear();

  if(res)
  {
    printf("Fail to parse %s\n", filename.c_str());
    exit(1);
  }
}

int
dbLefReader::lefUnitsCbk(lefrCallbackType_e c, lefiUnits* unit, lefiUserData ud)
{
  dbTech* tech = (dbTech*) ud;
  tech->setUnits(unit);
  return 0;
}

int
dbLefReader::lefBusBitCbk(lefrCallbackType_e c, const char* busBit, lefiUserData ud)
{
  dbTech* tech = (dbTech*) ud;
  tech->setBusBit(busBit);
  return 0;
}

int 
dbLefReader::lefDividerCbk(lefrCallbackType_e c, const char* div, lefiUserData ud)
{
  dbTech* tech = (dbTech*) ud;
  tech->setDivider(div[0]);
  return 0;
}

int 
dbLefReader::lefLayerCbk(lefrCallbackType_e c, lefiLayer* la, lefiUserData ud) 
{
  dbTech* tech = (dbTech*) ud;
  tech->createNewLayer(la);
  return 0;
}

int
dbLefReader::lefSiteCbk(lefrCallbackType_e c, lefiSite* si, lefiUserData ud ) 
{
  dbTech* tech = (dbTech*) ud;
  tech->createNewSite(si);
  return 0;
}

int 
dbLefReader::lefMacroBeginCbk(lefrCallbackType_e c, const char* name, lefiUserData ud ) 
{
  dbTech* tech = (dbTech*) ud;
  switch(c) 
  {
    case lefrMacroBeginCbkType:
      topMacro_ = tech->getNewMacro(name);
      break;
    default:
      break;
  }
  return 0;
}

int
dbLefReader::lefMacroCbk(lefrCallbackType_e c, lefiMacro* ma,  lefiUserData ud) 
{
  dbTech* tech = (dbTech*) ud;
  tech->fillNewMacro(ma, topMacro_);
  return 0;
}

int 
dbLefReader::lefMacroPinCbk(lefrCallbackType_e c, lefiPin* pi, lefiUserData ud ) 
{
  dbTech* tech = (dbTech*) ud;
  tech->addPinToMacro(pi, topMacro_);
  return 0; 
}

int 
dbLefReader::lefMacroObsCbk(lefrCallbackType_e c, lefiObstruction* obs, lefiUserData ud ) 
{
  dbTech* tech = (dbTech*) ud;
  tech->addObsToMacro(obs, topMacro_);
  return 0;
}

int
dbLefReader::lefViaCbk(lefrCallbackType_e c, lefiVia* via, lefiUserData ud)
{
  dbTech* tech = (dbTech*) ud;
	tech->createNewVia(via);
	return 0;
}

int 
dbLefReader::lefEndCbk(lefrCallbackType_e c, const char* name, lefiUserData ud ) 
{
  return 0;
}

}
