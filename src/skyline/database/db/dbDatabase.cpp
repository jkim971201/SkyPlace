#include <cassert>
#include <iostream>

#include "dbDatabase.h"

#include "dbTypes.h"
#include "dbTech.h"
#include "dbDesign.h"
#include "dbLefReader.h"
#include "dbDefReader.h"
#include "dbVerilogReader.h"
#include "dbBookShelfReader.h"

namespace db
{

dbDatabase::dbDatabase()
  : auxFile_       (""),
    vFile_         (""),
    defFile_       (""),
    bookShelfFlag_ (false)
{
  lefList_.clear();  

  // Initialization order matters.
  types_     = std::make_shared<dbTypes>();
  tech_      = std::make_shared<dbTech>(types_);
  design_    = std::make_shared<dbDesign>(types_, tech_);

  lefReader_     = std::make_shared<dbLefReader>(types_, tech_);
  defReader_     = std::make_shared<dbDefReader>(types_, tech_, design_);
  verilogReader_ = std::make_shared<dbVerilogReader>(tech_, design_);
  bsReader_      = std::make_shared<dbBookShelfReader>(types_, design_);
}

void
dbDatabase::readLef(const char* fileName)
{
  std::string filenameStr = std::string(fileName);

  if(lefList_.count(filenameStr))
    return;

  lefList_.insert(filenameStr);

  std::cout << "Read   " << filenameStr << std::endl;

  lefReader_->parseLef(filenameStr);

  std::cout << "Finish " << filenameStr << std::endl;
}

void
dbDatabase::readDef(const char* fileName)
{
  std::string filenameStr = std::string(fileName);

  if(defFile_ != "")
  {
    printf("You cannot read multiple .def files."); 
    printf(" %s will be ignored.\n", fileName); 
    return;
  }

  defFile_ = filenameStr;

  std::cout << "Read   " << filenameStr << std::endl;

  defReader_->parseDef(filenameStr);

  std::cout << "Finish " << filenameStr << std::endl;
}

void
dbDatabase::readVerilog(const char* fileName)
{
  std::string filenameStr = std::string(fileName);

  if(vFile_ != "")
  {
    printf("You cannot read multiple .v files.");
    printf(" %s will be ignored.\n", fileName);
    return;
  }

  vFile_ = filenameStr;

  std::cout << "Read   " << filenameStr << std::endl;

  verilogReader_->readFile(filenameStr);

  std::cout << "Finish " << filenameStr << std::endl;
}

void
dbDatabase::readBookShelf(const char* fileName)
{
  std::string filenameStr = std::string(fileName);

  if(auxFile_ != "")
  {
    printf("You cannot read multiple .aux files.");
    printf(" %s will be ignored.\n", fileName);
    return;
  }

  auxFile_ = filenameStr;

  std::cout << "Read   " << filenameStr << std::endl;

  bsReader_->readFile(filenameStr);
  tech_->setDbu(bsReader_->dbuBookShelf());

  bookShelfFlag_ = true;

  std::cout << "Finish " << filenameStr << std::endl;
}

void
dbDatabase::writeBookShelf(const char* filename) const
{
	design_->writeBookShelf(filename);
}

void
dbDatabase::writeDef(const char* filename) const
{
	design_->writeDef(filename);
}

void
dbDatabase::setTopModuleName(const char* topName)
{ 
  verilogReader_->setTopModuleName(std::string(topName));
}

}
