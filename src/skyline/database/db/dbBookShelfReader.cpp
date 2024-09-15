#include "dbBookShelfReader.h"
#include "dbMacro.h"
#include "dbLayer.h"
#include "dbRect.h"
#include "dbDie.h"
#include "dbRow.h"
#include "dbNet.h"
#include "dbInst.h"
#include "dbBTerm.h"
#include "dbITerm.h"

#include <map>
#include <cassert>

namespace db
{

dbBookShelfReader::dbBookShelfReader(std::shared_ptr<dbTypes>  types,
                                     std::shared_ptr<dbDesign> design)
  : types_        (types),
    design_       (design),
    dbuBookShelf_ (0)
{
  bsParser_ = std::make_unique<BookShelfParser>();
}

void
dbBookShelfReader::readFile(const std::string& filename)
{
  bsParser_->parse(filename);
  convert2db();
}

void
dbBookShelfReader::convert2db()
{
  // Note that BookShelf format does not have any
  // technology information.
  // Though some tech-related objects (e.g. dbMacro)
  // will be used in this function, but they are
  // just dummy objects to smoothly
  // convert bookshelf format to LEF/DEF-based database.
  const auto bookshelfDB = bsParser_->getDB();
  design_->setName(bsParser_->getBenchName()); 

  // IO pins of MMS benchmark are duplicate in .nets file.
  // A hash table is necessary to avoid multiple dbBTerm for
  // an identical IO pin.
  std::map<BsCell*, dbBTerm*> bsCell2dbBTerm;
  std::map<BsCell*, dbInst*>  bsCell2dbInst;

  // Bookshelf contains micron unit
  // so we need a scaling factor to
  // convert to integer data of dbInst, dbDie, ...
  // This is just a magic number.
  // (but this has be an even number because
  // all the numbers of bookshelf files are
  // multiplies of 0.5)
  constexpr int dbuBookShelf = 2;
  assert(dbuBookShelf >= 2);
  dbuBookShelf_ = dbuBookShelf;

  auto convert2dbDie = [&] (BsDie* bsDie)
  {
    auto dbDiePtr = design_->getDie();
    dbDiePtr->setLx(dbuBookShelf * bsDie->lx());
    dbDiePtr->setLy(dbuBookShelf * bsDie->ly());
    dbDiePtr->setUx(dbuBookShelf * bsDie->ux());
    dbDiePtr->setUy(dbuBookShelf * bsDie->uy());

    // Asuume BookShelf Format always have same core size with die
    design_->setCoreLx(dbuBookShelf * bsDie->lx());
    design_->setCoreLy(dbuBookShelf * bsDie->ly());
    design_->setCoreUx(dbuBookShelf * bsDie->ux());
    design_->setCoreUy(dbuBookShelf * bsDie->uy());
  };

  // This will be used to determine Macro
  const int rawRowHeight = bookshelfDB->rowHeight(); 
  
  int numRow = 0;
  auto convert2dbRow = [&] (BsRow* bsRow)
  {
    dbRow* newRow = new dbRow;
    const std::string row_name = "BookShelfRow" + std::to_string(numRow++);
    newRow->setName(row_name);
    newRow->setOrigX(dbuBookShelf * bsRow->lx());
    newRow->setOrigY(dbuBookShelf * bsRow->ly());
    newRow->setSiteSize(dbuBookShelf * bsRow->siteWidth(),  
                        dbuBookShelf * bsRow->dy());
    newRow->setStepX(dbuBookShelf * bsRow->siteSpacing());
    newRow->setStepY(dbuBookShelf * bsRow->dy());
    newRow->setNumSiteX(bsRow->numSites());
    newRow->setNumSiteY(1);
    return newRow;
  };

  // These dbMacros will not be registered to dbTech.
  // These are only to avoid dbInst making segmentation fault
  // due to dbInst methods that use dbMacro pointer.
  std::map<std::pair<int, int>, dbMacro*> size2macro;

  auto convert2dbInst = [&] (BsCell* bsCell)
  {
    dbInst* newInst = new dbInst;
    newInst->setName(bsCell->name());
    int macroSizeX = dbuBookShelf * bsCell->dx();
    int macroSizeY = dbuBookShelf * bsCell->dy();
    
    std::pair<int, int> macroSize = {macroSizeX, macroSizeY};

    dbMacro* macroPtr;
    auto itr = size2macro.find(macroSize);
    if(itr == size2macro.end())
    {
      macroPtr = new dbMacro;
      const std::string macro_name 
        = "BookShelfMacro" + std::to_string(size2macro.size());
      macroPtr->setName(macro_name.c_str());
      macroPtr->setSizeX(macroSizeX);
      macroPtr->setSizeY(macroSizeY);
      size2macro[macroSize] = macroPtr;
    }
    else
      macroPtr = itr->second;
  
    // Use raw value
    if(bsCell->dy() > rawRowHeight)
      macroPtr->setMacroClass(MacroClass::BLOCK);

    newInst->setMacro(macroPtr);

    newInst->setLocation(dbuBookShelf * bsCell->lx(), 
                         dbuBookShelf * bsCell->ly());

    if(bsCell->isFixed())
      newInst->setStatus(Status::FIXED);

    return newInst;
  };

  // dummy layer to make dbRect
  dbLayer* dummyLayer = new dbLayer; 
  auto convert2dbBTerm = [&] (BsCell* bsCell)
  {
    dbBTerm* newIO = new dbBTerm;

    dbBTermPort* newBTermPort = new dbBTermPort;
    newBTermPort->setOrigX(0);
    newBTermPort->setOrigY(0);  

    // Default orient is N
    newBTermPort->setLx( dbuBookShelf * bsCell->lx() );
    newBTermPort->setLy( dbuBookShelf * bsCell->ly() );
    newBTermPort->setUx( dbuBookShelf * bsCell->ux() );
    newBTermPort->setUy( dbuBookShelf * bsCell->uy() );
    newBTermPort->setLayer( dummyLayer );
    newIO->addPort( newBTermPort );
    return newIO;
  };

  // These dbMTerms will not be registered to dbTech.
  // These are only to avoid dbITerm making segmentation fault
  // (just same as dbMacros above)
  std::map<dbMacro*, std::map<std::pair<int, int>, dbMTerm*>> mterm_table;

  auto& dbITermVector = design_->getITerms();

  // To initialize map of map
  for(auto& kv : size2macro)
    mterm_table.insert(std::make_pair(kv.second, std::map<std::pair<int, int>, dbMTerm*>()));

  auto convert2dbNet = [&] (BsNet* bsNet)
  {
    dbNet* newNet = new dbNet;
    newNet->setName(bsNet->name());

    for(auto bsPinPtr : bsNet->pins())
    {
      BsCell* bsCellPtr = bsPinPtr->cell();

      auto bterm_itr = bsCell2dbBTerm.find(bsCellPtr);
      if(bterm_itr == bsCell2dbBTerm.end())
      {
        dbITerm* newITerm = new dbITerm;
        newITerm->setNet(newNet);

        // Set dbInst
        auto inst_itr = bsCell2dbInst.find(bsCellPtr);
        dbInst* inst_ptr;
        if(inst_itr == bsCell2dbInst.end())
        {
          printf("Error while converting bookshelf to db...\n");
          exit(1);
        }
        else
        {
          inst_ptr = inst_itr->second;
          const std::string iterm_name 
            = bsCellPtr->name() + std::to_string(inst_ptr->getITerms().size());
          newITerm->setName(iterm_name);
          newITerm->setInst(inst_ptr);
        }

        // Set dbMTerm
        dbMacro* macro_ptr = inst_ptr->macro();

        auto& offset2dbMTerm = mterm_table[macro_ptr];
        int offsetX = dbuBookShelf * bsPinPtr->offsetX();
        int offsetY = dbuBookShelf * bsPinPtr->offsetY();
        std::pair<int, int> offset = {offsetX, offsetY};

        dbMTerm* mterm;
        auto mterm_itr = offset2dbMTerm.find(offset);
        if(mterm_itr == offset2dbMTerm.end())
        {
          mterm = new dbMTerm;
  
          dbMTermPort* newPort = new dbMTermPort;
          newPort->addPoint(offsetX, offsetY);
          // POLYGON implicitly finish with the starting point.
          mterm->addPort( newPort );
          mterm->setBoundary();
          const std::string mterm_name 
           = macro_ptr->name() + "_" + std::to_string(macro_ptr->getMTerms().size());
          mterm->setName(mterm_name);
          macro_ptr->addMTerm(mterm);
          offset2dbMTerm[offset] = mterm;
        }
        else
          mterm = mterm_itr->second;

        newITerm->setMTerm(mterm);

        inst_ptr->addITerm(newITerm); 
        // addIterm checks if dbMTerm exists in dbITerm.
        // so this must be called after assign dbMTerm of this ITerm.

        // Finish ITerm
        newNet->addITerm(newITerm);
        dbITermVector.push_back(newITerm); 
        // new dbITerm should be added to dbDatabse
      }
      else
      {
        dbBTerm* bterm = bterm_itr->second;
        newNet->addBTerm(bterm);
      }
    }

    return newNet;
  };

  // Die
  convert2dbDie(bookshelfDB->getDie());

  // Row
  auto& dbRowVector = design_->getRows();
  auto& bsRowVector = bookshelfDB->rowVector();
  for(auto bsRowPtr : bsRowVector)
    dbRowVector.push_back(convert2dbRow(bsRowPtr));

  // Inst && BTerm (Bookshelf describes IOs as same as instance)
  auto& dbInstVector  = design_->getInsts();
  auto& dbBTermVector = design_->getBTerms();
  auto& bsCellVector = bookshelfDB->cellVector();
  for(auto bsCellPtr : bsCellVector)
  {
    // Bookshelf format does not have IOs explicitly.
    // we have to detect them by context.
    // (I think "terminal" keyword does not mean it's IO pin.)
    if(bsCellPtr->dx() == 0 || bsCellPtr->dy() == 0 || 
       (bsCellPtr->isFixed() && bsParser_->isOutsideDie(bsCellPtr)) )
    {
      auto newBTerm = convert2dbBTerm(bsCellPtr);
      dbBTermVector.push_back(newBTerm);
      bsCell2dbBTerm[bsCellPtr] = newBTerm;
    }
    else
    {
      auto newInst = convert2dbInst(bsCellPtr);
      dbInstVector.push_back(newInst);
      bsCell2dbInst[bsCellPtr] = newInst;
    }
  }

  // Net
  auto& bsNetVector = bookshelfDB->netVector();
  auto& dbNetVector = design_->getNets();
  for(auto bsNetPtr : bsNetVector)
  {
    auto newNet = convert2dbNet(bsNetPtr);
    dbNetVector.push_back(newNet);
  }
  
  printf("  Finish DB converting\n");
}

}
