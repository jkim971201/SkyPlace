#include <cstdio>
#include <cassert>
#include <cstdio>
#include <string>

#include "dbDefReader.h"
#include "dbUtil.h"

namespace db
{

static void defLogFunction(const char* errMsg) 
{
  printf("ERROR: %s\n", errMsg);
}

void checkType(defrCallbackType_e c)
{
  if(c >= 0 && c <= defrDesignEndCbkType) 
  {
    // OK
    // Do Nothing
  } 
  else 
    printf("ERROR: callback type is out of bounds!\n");
}

dbDefReader::dbDefReader(std::shared_ptr<dbTypes>  types, 
                         std::shared_ptr<dbTech>   tech,
                         std::shared_ptr<dbDesign> design)
  : types_   ( types),
    tech_    (  tech),
    design_  (design)
{
}

void
dbDefReader::init()
{
  defrSetLogFunction(defLogFunction);

  defrInitSession(0);
  
  defrSetUserData(design_.get());

  defrSetDesignCbk(this->defDesignCbk);

  // Unit
  defrSetUnitsCbk(this->defUnitsCbk);
  
  // Divider
  defrSetDividerCbk(this->defDividerCbk);

  // Die
  defrSetDieAreaCbk((defrBoxCbkFnType)this->defDieAreaCbk);

  // Rows 
  defrSetRowCbk((defrRowCbkFnType)this->defRowCbk);

  // Components
  defrSetComponentStartCbk(this->defComponentStartCbk);
  defrSetComponentCbk(this->defComponentCbk);
  defrSetComponentEndCbk(this->defComponentEndCbk);

  // Pins
  defrSetStartPinsCbk(this->defPinStartCbk);
  defrSetPinCbk((defrPinCbkFnType)this->defPinCbk);
  defrSetPinEndCbk(this->defPinEndCbk);

  // Nets
  defrSetNetStartCbk(this->defNetStartCbk);
  defrSetNetCbk(this->defNetCbk);
  defrSetNetEndCbk(this->defNetEndCbk);
  defrSetAddPathToNet(); 
  // Without this, routing information will be ignored.

  // Special Nets
  defrSetSNetStartCbk(this->defSNetStartCbk);
  defrSetSNetCbk(this->defSNetCbk);
  defrSetSNetEndCbk(this->defSNetEndCbk);

  // Generated Vias
  defrSetViaCbk(this->defViaCbk);

  // NonDefault Rules
  defrSetNonDefaultCbk(this->defNonDefaultRuleCbk);

  // Blockage
  defrSetBlockageCbk(this->defBlockageCbk);

  // Track
  defrSetTrackCbk(this->defTrackCbk);

  // End Design
  defrSetDesignEndCbk(this->defEndCbk);
}

void
dbDefReader::parseDef(const std::string& filename)
{
  init();

  size_t dot = filename.find_last_of('.');
  std::string filetype = filename.substr(dot + 1);

  if(filetype != "def")
  {
    printf("Please give .def file!\n");
    exit(1);
  }

  FILE* file = fopen(filename.c_str(), "r");

  if(file == nullptr)
  {
    printf("Cannot open %s\n", filename.c_str());
    exit(1);
  }

  int res = defrRead(file, filename.c_str(), (void*) design_.get(), 1);
  fclose(file);

  defrClear();

  if(res)
  {
    printf("Fail to parse %s\n", filename.c_str());
    exit(1);
  }
}

// Designs
int 
dbDefReader::defDesignCbk(defrCallbackType_e c, const char* name, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->setName(name);
  return 0;
}

// Units
int 
dbDefReader::defUnitsCbk(defrCallbackType_e c, double unit, defiUserData ud)
{
  checkType(c);
  
  int defDbu = static_cast<int>(unit);
  dbDesign* design = (dbDesign*) ud;
  design->setDbu(defDbu);

  return 0;
}

// Units
int 
dbDefReader::defDividerCbk(defrCallbackType_e c, const char* div, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->setDivider(div[0]);
  return 0;
}

// Die
int 
dbDefReader::defDieAreaCbk(defrCallbackType_e c, defiBox* box, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->setDie(box);
  return 0;
}

// Rows
int 
dbDefReader::defRowCbk(defrCallbackType_e c, defiRow* ro, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->addNewRow(ro);
  return 0;
}

// Nets
int 
dbDefReader::defNetStartCbk(defrCallbackType_e c, int number, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;

  if(design->getNets().size() > number)
    assert(0);
  else
    design->getNets().reserve(number);
  return 0;
}

int 
dbDefReader::defNetCbk(defrCallbackType_e c, defiNet* net, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  
  const std::string& nameWithoutBackSlash
    = removeBackSlashBracket( std::string(net->name()) );

  dbNet* dbnet = design->getNetByName(nameWithoutBackSlash);

  if(dbnet == nullptr)
    dbnet = design->getNewNet(nameWithoutBackSlash);

  design->fillNet(net, dbnet);

  return 0;
}

int
dbDefReader::defNetEndCbk(defrCallbackType_e c, void* , defiUserData ud)
{
  checkType(c);
  return 0;
}
    
// Special Nets
int 
dbDefReader::defSNetStartCbk(defrCallbackType_e c, int number, defiUserData ud)
{
  checkType(c);
  return 0;
}

int 
dbDefReader::defSNetCbk(defrCallbackType_e c, defiNet* net, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->addNewSNet(net);
  return 0;
}

int 
dbDefReader::defSNetEndCbk(defrCallbackType_e c, void* , defiUserData ud)
{
  checkType(c);
  return 0;
}

// Pins
int 
dbDefReader::defPinStartCbk(defrCallbackType_e c, int  number, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;

  if(design->getBTerms().size() > number)
    assert(0);
  else
    design->getBTerms().reserve(number);
  return 0;
}

int 
dbDefReader::defPinCbk(defrCallbackType_e c, defiPin* pi, defiUserData ud)
{
  checkType(c);
  
  dbDesign* design = (dbDesign*) ud;

  std::string nameWithoutBackSlash = removeBackSlashBracket( std::string(pi->pinName()) );
  design->addNewIO(pi, nameWithoutBackSlash);
  return 0;
}

int 
dbDefReader::defPinEndCbk(defrCallbackType_e c, void*  , defiUserData ud)
{
  checkType(c);
  return 0;
}

// Components
int 
dbDefReader::defComponentStartCbk(defrCallbackType_e c, int number, defiUserData ud)
{
  checkType(c);

  dbDesign* design = (dbDesign*) ud;

  if(design->getInsts().size() > number)
    assert(0);
  else
    design->getInsts().reserve(number);

  return 0;
}

int 
dbDefReader::defComponentCbk(defrCallbackType_e c, defiComponent* co, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
 
  std::string nameWithoutBackSlash = removeBackSlashBracket( std::string(co->id()) );

  dbInst* inst = design->getInstByName( nameWithoutBackSlash );

  if(inst == nullptr)
    design->addNewInst(co, nameWithoutBackSlash);
  else
    design->fillInst(co, inst);

  return 0;
}

int 
dbDefReader::defComponentEndCbk(defrCallbackType_e c, void* , defiUserData ud)
{
  checkType(c);
  return 0;
}

int 
dbDefReader::defViaCbk(defrCallbackType_e c, defiVia* via, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->addNewViaMaster(via);
  return 0;
}


int 
dbDefReader::defNonDefaultRuleCbk(defrCallbackType_e c, defiNonDefault* ndr, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->addNewNonDefaultRule(ndr);
  return 0;
}

int 
dbDefReader::defBlockageCbk(defrCallbackType_e c, defiBlockage* blk, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->addNewBlockage(blk);
  return 0;
}

int 
dbDefReader::defTrackCbk(defrCallbackType_e c, defiTrack * tr, defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->addNewTrack(tr);
  return 0;
}

// End Design
int 
dbDefReader::defEndCbk(defrCallbackType_e c, void* , defiUserData ud)
{
  checkType(c);
  dbDesign* design = (dbDesign*) ud;
  design->finish(); // post-process
  return 0;
}

}
