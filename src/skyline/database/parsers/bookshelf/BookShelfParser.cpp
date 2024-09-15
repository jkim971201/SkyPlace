#include <iostream>
#include <cassert> // For Debug-Mode
#include <cstring>
#include <filesystem>
#include <stdio.h>
#include <string.h>

#include "BookShelfParser.h"

#define BUF_SIZE 512

namespace bookshelf
{

inline char* getNextWord(const char* delimiter)
{
  // strtok() has "static char *olds" inside
  // if strtok() gets nullptr, then it starts from the "olds"
  // so you can get the next word
  // you must use strtok(buf, ~) to capture the first word
  // otherwise, you will get segmentation fault
  // (this is implemented in goNextLine)
  return strtok(nullptr, delimiter);
}

// Go Next Line and Get First Word
inline char* goNextLine(char* buf, const char* delimiter, FILE* fp)
{
  fgets(buf, BUF_SIZE-1, fp);
  return strtok(buf, delimiter);
}

inline void extractDir(char* file, char* dir)
{
  int nextToLastSlash = 0;
  int len = strlen(file);

  strcpy(dir, file);

  for(int i = 0; i < len; i++) 
  {
    if(file[i] == '/') 
      nextToLastSlash = i + 1;
  }

  dir[nextToLastSlash] = '\0';
  // \0 == nullptr
}

// What a stupid way...
// But this seems best... at least to me...
inline void catDirName(char* dir, char* name)
{
  char temp[MAX_FILE_NAME];
  strcpy(temp, dir);
  strcat(temp, name);
  strcpy(name, temp);
}

inline void getOnlyName(const char* file, char* name)
{
  int lastDot = 0;
  int nextToLastSlash = 0;
  int len = strlen(file);

  for(int i = 0; i < len; i++) 
  {
    if(file[i] == '.') 
      lastDot = i;
    if(file[i] == '/') 
      nextToLastSlash = i + 1;
  }

  int nameLength = lastDot - nextToLastSlash;
  for(int i = 0; i < nameLength; i++)
    name[i] = file[i + nextToLastSlash];

  name[nameLength] = '\0';
}

inline void getSuffix(const char* token, char* sfx)
{
  int lastDot = 0;
  int len = strlen(token);

  for(int i = 0; i < len; i++) 
  {
    if(token[i] == '.') 
      lastDot = i;
  }
  strcpy(sfx, &token[lastDot + 1]);
}

// BsCell //
BsCell::BsCell()
{
  lx_ = ly_ = ux_ = uy_ = 0;
  id_ = 0;
  orient_ = 'N';
  isFixed_ = isFixedNI_ = false;
}

BsCell::BsCell(std::string& name,
               int  id,
               int  width, 
               int  height, 
               bool isTerminal, 
               bool isTerminalNI) 
    : BsCell()
{
  name_ = name;

  id_ = id;
  dx_ = width;
  dy_ = height;
  isTerminal_   = isTerminal;
  isTerminalNI_ = isTerminalNI;
}

// BsRow //
BsRow::BsRow(int idx,
             int ly, 
             int rowHeight, 
             int siteWidth, 
             int siteSpacing,
             int offsetX,
             int numSites)
{
  idx_          = idx;

  // Explicit Values from .scl file
  ly_           = ly;

  rowHeight_    = rowHeight;
  siteWidth_    = siteWidth;
  siteSpacing_  = siteSpacing;
  offsetX_      = offsetX;
  numSites_     = numSites;
  siteOrient_   = true;
  siteSymmetry_ = true;

  // Implicit Value
  rowWidth_ = numSites_ * siteWidth_;
}

// BsPin //
BsPin::BsPin(BsCell* cell, int netID, 
             double offsetX, double offsetY,
             char IO)
{
  cell_  = cell;
  net_   = nullptr;

  netID_ = netID;

  offsetX_ = offsetX;
  offsetY_ = offsetY;

  io_ = IO;
}

// BookShelfDB //
BookShelfDB::BookShelfDB(int numNodes)
{
  numRows_    = 0;
  numFixed_   = 0;
  numFixedNI_ = 0;
  numMovable_ = 0;
  numInst_    = 0;
  rowHeight_  = 0;

  numBsCells_ = numNodes;
  cellPtrs_.reserve(numNodes);
  cellInsts_.reserve(numNodes);
}

void
BookShelfDB::makeBsCell(std::string& name, 
                        int  width, 
                        int  height, 
                        bool isTerminal, 
                        bool isTerminalNI)
{
  BsCell oneBsCell(name, numInst_, 
                   width, height, 
                   isTerminal, isTerminalNI);

  numInst_++;

  // TODO: Correct this
  if(isTerminal)        numFixed_++;
  else if(isTerminalNI) numFixedNI_++;
  else                  numMovable_++;

  cellInsts_.push_back(oneBsCell);
  // cellPtrs will be filled by buildBsCellMap()
}

void
BookShelfDB::buildBsCellMap()
{
  //printf("  Building Node Map\n");
  for(BsCell& c : cellInsts_)
  {
    cellPtrs_.push_back(&c);
    cellMap_.emplace(c.name(), &c);
  }
}

void
BookShelfDB::makeBsRow(int idx,
                       int ly, 
                       int rowHeight, 
                       int siteWidth, 
                       int siteSpacing,
                       int offsetX,
                       int numSites)
{
  BsRow oneBsRow(idx, ly, 
                 rowHeight, 
                 siteWidth, 
                 siteSpacing, 
                 offsetX, 
                 numSites);
  rowInsts_.push_back(oneBsRow);
  // rowPtrs will be filled by buildBsRowMap()
}

void
BookShelfDB::makeBsNet(int netID, const std::string& name)
{
  BsNet oneBsNet(netID);
	oneBsNet.setName(name);
  netInsts_.push_back(oneBsNet);
  // netPtrs will be filled by finishPinsAndNets()
}

void
BookShelfDB::makeBsPin(BsCell* cell, int netID, 
                       double offsetX, double offsetY,
                       char IO)
{
  double cell_half_w = static_cast<double>(cell->dx()) / 2;
  double cell_half_h = static_cast<double>(cell->dy()) / 2;

  if(std::abs(offsetX) > cell_half_w)
  {
    printf("  [Warning] Pin of Cell %s is out of Cell Boundary.\n", 
              cell->name().c_str());
    printf("  [Warning] Pin OffsetX %.1f is larger than Cell Half Width %.1f \n",
              offsetX, cell_half_w);
  }

  if(std::abs(offsetY) > cell_half_h)
  {
    printf("  [Warning] Pin of Cell %s is out of Cell Boundary.\n", 
              cell->name().c_str());
    printf("  [Warning] Pin OffsetY %.1f is larger than Cell Half Height %.1f \n",
              offsetY, cell_half_h);
  }

  BsPin oneBsPin(cell, netID, offsetX, offsetY, IO);

  pinInsts_.push_back(oneBsPin);
}

void
BookShelfDB::finishPinsAndNets()
{
  //printf("Building Net maps.\n");

  for(auto& net : netInsts_)
  {
    netPtrs_.push_back(&net); 
    netMap_[net.id()] = &net; 
  }

  //printf("  Adding pins to cells and nets.\n");

  for(auto& pin : pinInsts_)
  {
    pinPtrs_.push_back(&pin);
    pin.cell()->addNewPin(&pin);
    if(!pin.net()) 
      pin.setNet(getBsNetByID(pin.netID()));
    else
    {
      printf("  Unknown Error...\n");
      exit(0);
    }

    if(pin.net()) 
      pin.net()->addNewPin(&pin);
    else
    {
      printf("  Unknown Error...\n");
      exit(0);
    }
  }
}

void
BookShelfDB::buildBsRowMap()
{
  int maxX = 0;
  int maxY = 0;

  int minX = std::numeric_limits<int>::max();
  int minY = std::numeric_limits<int>::max();

  //printf("  Building Row Map\n");
  for(BsRow& r : rowInsts_)
  {
    if(r.ux() > maxX) maxX = r.ux();
    if(r.lx() < minX) minX = r.lx();
    if(r.uy() > maxY) maxY = r.uy();
    if(r.ly() < minY) minY = r.ly();
    rowPtrs_.push_back(&r);
    rowMap_.emplace(r.id(), &r);
  }

  //for(auto& c : cellPtrs_)
  //  if(c->uy() < minY) minY = c->uy();

  bsDie_.setUxUy(maxX, maxY);
  bsDie_.setLxLy(minX, minY);
  bsDiePtr_ = &bsDie_;

  //printf("Creating a Die (%d, %d) - (%d, %d) \n", maxX, maxY, minX, minY);
  numRows_ = rowPtrs_.size();
}

void
BookShelfDB::verifyMap()
{
  std::cout << "Start Verifying Map Vector" << std::endl;
  for(auto kv : cellMap_)
  {
    std::cout << "key name: " << kv.first << std::endl;
    std::cout << "ptr width: " << kv.second->dx() << std::endl;
  }
}

void
BookShelfDB::verifyVec()
{
  std::cout << "Start Verifying Instance Vector" << std::endl;
  for(auto c : cellInsts_)
  {
    std::cout << "cell name: " << c.name() << std::endl;
    std::cout << "cell width: " << c.dx() << std::endl;
  }
}

void
BookShelfDB::verifyPtrVec()
{
  std::cout << "Start Verifying Pointer Vector" << std::endl;
  for(int i = 0; i < cellPtrs_.size(); i++)
  {
    std::cout << "cell name: " << cellPtrs_[i]->name() << std::endl;
    std::cout << "cell width: " << cellPtrs_[i]->dx() << std::endl;
  }
}

BookShelfParser::BookShelfParser()
{
  numFixed_     = 0;
  numFixedNI_   = 0;
  numMovable_   = 0;
  maxRowHeight_ = 0;
  bookShelfDB_  = nullptr;
}

void
BookShelfParser::init(const char* aux_name)
{
  char suf[4];
  getSuffix(aux_name, suf);

  getOnlyName(aux_name, benchName_);

  if(!strcmp(suf, "aux"))
  {
    strcpy(aux_, aux_name);
    extractDir(aux_, dir_);
  }
  else
  {
    printf("Make sure you give .aux file\n");
    exit(0);
  }
}

void
BookShelfParser::parse(const std::filesystem::path& aux_name)
{
  if(!std::filesystem::exists(aux_name))
  {
    printf("aux file %s not exists!\n");
    exit(1);
  }

  // filetype check will be done in BookShelfParser::init
  std::string filename = std::string(aux_name);
  init(filename.c_str());

  // Start from .aux file
  read_aux();
  read_nodes();
  read_pl();
  read_scl();
  read_nets();

  //printf(" Parsing is finished successfully!\n");
}

void
BookShelfParser::read_aux()
{
  //printf("  Reading %s...\n", aux_);
  FILE *fp = fopen(aux_, "r");
  char *token = nullptr;
  char buf[BUF_SIZE-1];

  char sfx[6]; // the longest is "nodes" (5 letters)

  if(fp == nullptr)
  {
    printf(" Failed to open %s...\n", aux_);
    exit(0);
  }
  
  token = goNextLine(buf, " :", fp);

  if(strcmp(token, "RowBasedPlacement"))
  {
    printf(" Unknown Placement Type: %s\n", token);
    exit(0);
  }

  while(true)
  {
    token = getNextWord(" :\n");
    if(!token) break;  

    getSuffix(token, sfx);

    if(!strcmp(sfx, "nodes"))
    {
     //printf(" .nodes file detected.\n");
     strcpy(nodes_, token);
     catDirName(dir_, nodes_);
    }
    else if(!strcmp(sfx, "pl"))
    {
     //printf(" .pl file detected.\n");
     strcpy(pl_, token);
     catDirName(dir_, pl_);
    }
     else if(!strcmp(sfx, "scl"))
    {
      //printf(" .scl file detected.\n");
      strcpy(scl_, token);
      catDirName(dir_, scl_);
    }
    else if(!strcmp(sfx, "nets"))
    {
      //printf(" .nets file detected.\n");
      strcpy(nets_, token);
      catDirName(dir_, nets_);
    }
    else if(!strcmp(sfx, "wts"))
    {
      printf("  .wts file is not supported. (will be ignored)\n");
    }
    else
    {
      printf("Unknown file format %s...\n", sfx);
			exit(1);
    }
  }

  fclose(fp);
}

void
BookShelfParser::read_nodes()
{
  printf("  Reading %s...\n", nodes_);

  FILE *fp = fopen(nodes_, "r");
  char *token = nullptr;
  char buf[BUF_SIZE-1];

  if(fp == nullptr)
  {
    printf("  Failed to open %s...\n", nodes_);
    exit(0);
  }

  // Skip Headlines
  while(!token || !strcmp(token, "UCLA") || token[0] == '#')
    token = goNextLine(buf, " \t\n", fp);

  // Read NumNodes (at this moment, buf == "NumNodes")
  assert(!strcmp(buf, "NumNodes"));
  token = getNextWord(" \t\n");
  token = getNextWord(" \t\n");
  int numNodes = atoi(token);
  token = goNextLine(buf, " \t\n", fp);
  
  // Read NumTerminals (at this moment, buf == "NumTerminals")
  assert(!strcmp(buf, "NumTerminals"));
  token = getNextWord(" \t\n");
  token = getNextWord(" \t\n");
  int numTerminals = atoi(token);
  token = getNextWord(" \t\n");
  //printf(" Total Nodes: %d\n", numNodes);
  //printf(" Total Terms: %d\n", numTerminals);

  // Go to Next Line untill there are no blank lines anymore
  while(!token)
    token = goNextLine(buf, " \t\n", fp);

  bookShelfDB_ = std::make_shared<BookShelfDB>(numNodes);

  int numLines = 0;

  int width  = 0;
  int height = 0;
  bool isTerminal   = false;
  bool isTerminalNI = false;

  while(!feof(fp))
  {
    isTerminal   = false;
    isTerminalNI = false;
    std::string cellName = std::string(token);

    // Get Width
    token = getNextWord(" \t");
    width = atoi(token);

    // Get Height
    token = getNextWord(" \t");
    height = atoi(token);

    // Check Terminal
    token = getNextWord(" \t");
    if(token && !strcmp(token, "terminal\n")) 
      isTerminal   = true;
    else if(token && !strcmp(token, "terminal_NI\n"))
      isTerminalNI = true;

    // Make a BsCell (==Node)
    bookShelfDB_->makeBsCell(cellName, width, height, isTerminal, isTerminalNI);
    // printf(" CellName: %s Width: %d Height: %d\n", cellName.c_str(), width, height); 
    numLines++;

    //if(numLines % 100000 == 0)
    //  printf(" Completed %d lines\n", numLines);

    token = goNextLine(buf, " \t\n", fp);
  }
  fclose(fp);

  bookShelfDB_->buildBsCellMap();

  // For Debug
  assert(numNodes == bookShelfDB_->cellVector().size());
  assert(numLines == bookShelfDB_->cellVector().size());

  //printf("Successfully Finished %s!\n", nodes_);
}

void
BookShelfParser::read_pl()
{
  printf("  Reading %s...\n", pl_);

  FILE *fp = fopen(pl_, "r");
  char *token = nullptr;
  char buf[BUF_SIZE-1];

  if(fp == nullptr)
  {
    printf("  Failed to open %s...\n", pl_);
    exit(0);
  }

  // Skip Headlines
  while(!token || !strcmp(token, "UCLA") || token[0] == '#')
    token = goNextLine(buf, " \t\n", fp);

  // Go to Next Line untill there are no blank lines anymore
  while(!token)
    token = goNextLine(buf, " \t\n", fp);

  int numLines = 0;

  int lx = 0;
  int ly = 0;

  char orient;

  bool isFixed   = false;
  bool isFixedNI = false;

  while(!feof(fp))
  {
    std::string cellName = std::string(token);

    BsCell* myBsCell = bookShelfDB_->getBsCellByName(cellName);
    assert(cellName == myBsCell->name());

    // Get X Coordinate
    token = getNextWord(" \t");
    lx = atoi(token);

    // Get Y Coordinate
    token = getNextWord(" \t");
    ly = atoi(token);

    myBsCell->setXY(lx, ly);

    // Get Orient
    token = getNextWord(" \t\n:");
    orient = atoi(token);

    if(orient != 'N') 
      myBsCell->setOrient(orient);

    // Get Move-Type
    token = getNextWord(" \t");
    if(token && !strcmp(token, "/FIXED\n")) 
    {
      myBsCell->setFixed();
      numFixed_++;
    }
    else if(token && !strcmp(token, "/FIXED_NI\n"))
    {
      myBsCell->setFixedNI();
      numFixedNI_++;
    }
    else
    {
      numMovable_++;
    }

    //printf(" %s: %d %d\n", cellName.c_str(), lx, ly);
  
    numLines++;

    //if(numLines % 100000 == 0)
    //  printf("  Completed %d lines\n", numLines);

    token = goNextLine(buf, " \t\n", fp);
  }

  assert(numLines == bookShelfDB_->cellVector().size());
  //printf("  Successfully Finished %s!\n", pl_);
}


void
BookShelfParser::read_scl()
{
  printf("  Reading %s...\n", scl_);

  FILE *fp = fopen(scl_, "r");
  char *token = nullptr;
  char buf[BUF_SIZE-1];

  if(fp == nullptr)
  {
    printf("  Failed to open %s...\n", scl_);
    exit(0);
  }

  // Skip Headlines
  while(!token || !strcmp(token, "UCLA") || token[0] == '#')
    token = goNextLine(buf, " \t\n", fp);

  // Read NumRows (at this moment, buf == "NumRows")
  assert(!strcmp(buf, "NumRows") || !strcmp(buf, "Numrows"));
  token = getNextWord(" \t\n");
  token = getNextWord(" \t\n");
  int numRows = atoi(token);
  token = getNextWord(" \t\n");

  // Go to Next Line untill there are no blank lines anymore
  while(!token)
    token = goNextLine(buf, " \t\n", fp);

  int rowsRead = 0;
  int maxRowHeight = 0;

  while(!feof(fp))
  {
    int  ly;           
    int  rowHeight;    
    int  siteWidth;    
    int  siteSpacing;  
    char siteOrient;   
    char siteSymmetry;
    int  offsetX;     
    int  numSites;     
   
    // Read a Row (at this moment, token == "CoreRow")
    assert(!strcmp(token, "CoreRow"));

    while(true)
    {
      token = goNextLine(buf, " \t\n:", fp); 
      // in the initial step of this loop 
      // token == "Coordinate" at this moment

      if(!strcmp(token, "Coordinate"))
      {
        token = getNextWord(" \t\n:");
        ly = atoi(token);
      }
      else if(!strcmp(token, "Height"))
      {
        token = getNextWord(" \t\n:");
        rowHeight = atoi(token);
        if(rowHeight > maxRowHeight)
          maxRowHeight = rowHeight;
      }
      else if(!strcmp(token, "Sitewidth"))
      {
        token = getNextWord(" \t\n:");
        siteWidth = atoi(token);
      }
      else if(!strcmp(token, "Sitespacing"))
      {
        token = getNextWord(" \t\n:");
        siteSpacing = atoi(token);
      }
      else if(!strcmp(token, "Siteorient"))
      {
        token = getNextWord(" \t\n:");
        siteOrient = token[0];
      }
      else if(!strcmp(token, "Sitesymmetry"))
      {
        token = getNextWord(" \t\n:");
        siteOrient = token[0];
      }
      else if(!strcmp(token, "SubrowOrigin"))
      {
        token = getNextWord(" \t\n:");
        offsetX = atoi(token);
        token = getNextWord(" \t\n:");
        if(!strcmp(token, "NumSites") || !strcmp(token, "Numsites"))
        {
          token = getNextWord(" \t\n:");
          numSites = atoi(token);
        }
        else
        {
          printf("  Wrong BookShelf Syntax\n");
          printf("  Current Token %s\n", token);
          exit(0);
        }
      }
      else if(!strcmp(token, "End"))
      {
        bookShelfDB_->makeBsRow(rowsRead, // idx
                              ly, rowHeight, 
                              siteWidth, siteSpacing, 
                              offsetX, numSites);
        rowsRead++;
        token = goNextLine(buf, " \t\n:", fp); 
        break;
      }
      else
      {
        printf("  Wrong BookShelf Syntax\n");
        printf("  Current Token %s\n", token);
        exit(0);
      }
    }

    //if(rowsRead % 500 == 0)
    //  printf("Completed %d rows\n", rowsRead);

    if(rowsRead > numRows) 
    {
      printf("  Extra Rows more than %d will be ignored...\n", numRows);
      break;
    }
  }

  bookShelfDB_->buildBsRowMap();
  bookShelfDB_->setHeight(maxRowHeight);
  assert(numRows == bookShelfDB_->rowVector().size());
  //printf("  Successfully Finished %s!\n", scl_);
}

void
BookShelfParser::read_nets()
{
  printf("  Reading %s...\n", nets_);

  FILE *fp = fopen(nets_, "r");
  char *token = nullptr;
  char buf[BUF_SIZE-1];

  if(fp == nullptr)
  {
    printf("Failed to open %s...\n", nets_);
    exit(0);
  }

  // Skip Headlines
  while(!token || !strcmp(token, "UCLA") || token[0] == '#')
    token = goNextLine(buf, " \t\n", fp);

  // Read NumNets (at this moment, buf == "NumNets")
  assert(!strcmp(buf, "NumNets"));
  token = getNextWord(" \t\n:");
  int numNets = atoi(token);
  token = goNextLine(buf, " \t\n", fp);

  // Read NumPins (at this moment, buf == "NumPins")
  assert(!strcmp(buf, "NumPins"));
  token = getNextWord(" \t\n:");
  int numPins = atoi(token);
  token = getNextWord(" \t\n");
  token = goNextLine(buf, " \t\n", fp);

  //printf(" Total Nets: %d\n", numNets);
  //printf(" Total Pins: %d\n", numPins);

  // Go to Next Line untill there are no blank lines anymore
  while(!token)
    token = goNextLine(buf, " \t\n", fp);

  int netsRead = 0;

  while(!feof(fp))
  {
    // Read a Net (at this moment, token == "NetDegree")
    assert(!strcmp(token, "NetDegree"));
    token = getNextWord(" \t\n:");
    int netDegree = atoi(token);

    token = getNextWord(" \t\n");
    std::string netName = std::string(token);

    // Some nets have NetDegree : 1 in MMS benchmarks
		// Ignore these NetDegree 1 nets
		if(netDegree > 1)
      bookShelfDB_->makeBsNet(netsRead, netName); // netsRead => netID

    char IO;
    double offsetX;
    double offsetY;

    int pinsRead = 0;
    while(pinsRead < netDegree)
    {
      // Get Master Cell's Name
      token = goNextLine(buf, " \t", fp);
      BsCell* cellOfThesePins = bookShelfDB_->getBsCellByName(std::string(token));

      // Get Pin IO Type
      token = getNextWord(" \t\n");
      IO = token[0];

      if(IO != 'I' && IO != 'O' && IO != 'B')
      {
        printf("  Wrong BookShelf Syntax.\n");
        printf("  Pin IO type must be one of I, O, B.\n");
        exit(0);
      }

      // Get Pin Offset X
      token = getNextWord(" \t\n:");
      if(token == nullptr) // In ICCAD 2004 Benchmarks,
        offsetX = 0.0;  // They do not provide offsets of Chip IOs
      else 
        offsetX = atof(token);

      // Get Pin Offset Y
      token = getNextWord(" \t\n:");
      if(token == nullptr) // In ICCAD 2004 Benchmarks,
        offsetY = 0.0;  // They do not provide offsets of Chip IOs
      else 
        offsetY = atof(token);

			// Ignore pins of Degree-1 net
			if(netDegree > 1)
			{
        bookShelfDB_->makeBsPin(cellOfThesePins, 
                                netsRead, // netID 
                                offsetX, offsetY, IO);
			}
      pinsRead++;
    }

    netsRead++;
    //if(netsRead % 100000 == 0)
    //  printf("  Completed %d nets\n", netsRead);

    token = goNextLine(buf, " \t\n", fp);
  }

  bookShelfDB_->finishPinsAndNets();
  //printf("Successfully Finished %s!\n", nets_);
}

// Temporary...
bool 
BookShelfParser::isOutsideDie(BsCell* cell)
{
  bool isOutside = false;

  BsDie* die = bookShelfDB_->getDie();

  int dieLx = die->lx();
  int dieLy = die->ly();
  int dieUx = die->ux();
  int dieUy = die->uy();

  if(cell->ux() <= dieLx) isOutside = true;
  if(cell->lx() >= dieUx) isOutside = true;
  if(cell->uy() <= dieLy) isOutside = true;
  if(cell->ly() >= dieUy) isOutside = true;

  return isOutside;
}

void
BookShelfParser::printInfo()
{
  using namespace std;

  string name = string(benchName_);

  int numInst = numFixed_ + numMovable_;
  int numNet  = bookShelfDB_->netVector().size();
  int numPin  = bookShelfDB_->pinVector().size();
  int numRow  = bookShelfDB_->rowVector().size();

  BsDie* die = bookShelfDB_->getDie();

  int dieLx = die->lx();
  int dieLy = die->ly();
  int dieUx = die->ux();
  int dieUy = die->uy();

  int dieArea = die->area();

  int sumTotalInstArea = 0;
  int sumMovableArea = 0;
  int sumFixedArea = 0;

  for(auto& cell : bookShelfDB_->cellVector() )
  {
    int area = cell->area();
    
    if( cell->isFixed() )
    {
      if( isOutsideDie(cell) )
        continue;
    }

    sumTotalInstArea += area;

    if( cell->isFixed() )
      sumFixedArea += area;
    else
      sumMovableArea += area;
  }

  double density = static_cast<double>(sumTotalInstArea) 
                 / static_cast<double>(dieArea);

  double util    = static_cast<double>(sumMovableArea)   
                 / static_cast<double>(dieArea - sumFixedArea);

  cout << endl;
  cout << "*** Summary of Information ***" << endl;
  cout << "---------------------------------------------" << endl;
  cout << " DESIGN INFO"                                  << endl;
  cout << "---------------------------------------------" << endl;
  cout << " DESIGN NAME      : " << name          << endl;
  cout << " NUM INSTANCE     : " << numInst       << endl;
  cout << " NUM MOVABLE      : " << numMovable()  << endl;
  cout << " NUM FIXED        : " << numFixed()    << endl;
  cout << " NUM NET          : " << numNet        << endl;
  cout << " NUM PIN          : " << numPin        << endl;
  cout << " NUM ROW          : " << numRow        << endl;
  cout << " UTIL             : " << fixed << setprecision(2) << util    * 100 << "%\n";
  cout << " DENSITIY         : " << fixed << setprecision(2) << density * 100 << "%\n";
  cout << " AREA (INSTANACE) : " << setw(10) << sumTotalInstArea << endl;
  cout << " AREA (MOVABLE)   : " << setw(10) << sumMovableArea   << endl;
  cout << " AREA (FIXED)     : " << setw(10) << sumFixedArea     << endl;
  cout << " AREA (CORE)      : " << setw(10) << dieArea          << endl;
  cout << " CORE ( " << setw(5) << dieLx  << " ";
  cout << setw(5) << dieLy  << " ) ( ";
  cout << setw(8) << dieUx  << " " << dieUy  << " )\n";
  cout << "---------------------------------------------" << endl;
}

} // namespace BookShelf
