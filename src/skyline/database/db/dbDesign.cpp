#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>

#include "dbDesign.h"
#include "dbUtil.h"

#include "dbTypes.h"
#include "dbTech.h"
#include "dbDie.h"
#include "dbRow.h"
#include "dbInst.h"
#include "dbNet.h"
#include "dbBTerm.h"
#include "dbITerm.h"
#include "dbWire.h"
#include "dbViaInst.h"
#include "dbViaMaster.h"
#include "dbNonDefaultRule.h"
#include "dbBlockage.h"
#include "dbTrackGrid.h"

namespace db
{

dbDesign::dbDesign(const std::shared_ptr<dbTypes> types,
                   const std::shared_ptr<dbTech>  tech)
  : tech_    (tech),
    types_   (types),
    name_    (""),
    divider_ ('/'),
    die_     (nullptr),
    coreLx_  (std::numeric_limits<int>::max()),
    coreLy_  (std::numeric_limits<int>::max()),
    coreUx_  (std::numeric_limits<int>::min()),
    coreUy_  (std::numeric_limits<int>::min())
{
}

dbDesign::~dbDesign()
{
  delete die_;

  rows_.clear();
  insts_.clear();
  nets_.clear();
  bterms_.clear();
  iterms_.clear();
  vias_.clear();

  str2dbInst_.clear();
  str2dbBTerm_.clear();
  str2dbNet_.clear();
}

void
dbDesign::finish()
{
  // #1. Sort tracks in the increasing order.
  for(auto trackGrid : trackGrids_)
    trackGrid->sortTracks();
}

dbInst*
dbDesign::getInstByName(const std::string& name)
{
  auto itr = str2dbInst_.find(name);
  
  if(itr == str2dbInst_.end())  
    return nullptr;
  else
    return itr->second;
}

dbBTerm*
dbDesign::getBTermByName(const std::string& name)
{  
  auto itr = str2dbBTerm_.find(name);
  
  if(itr == str2dbBTerm_.end())  
    return nullptr;
  else
    return itr->second;
}

dbNet*
dbDesign::getNetByName(const std::string& name)
{ 
  auto itr = str2dbNet_.find(name);
  
  if(itr == str2dbNet_.end())  
    return nullptr;
  else
    return itr->second;
}

dbViaMaster*
dbDesign::getViaMasterByName(const std::string& name)
{ 
  auto itr = str2dbViaMaster_.find(name);
  
  if(itr == str2dbViaMaster_.end())  
    return nullptr;
  else
    return itr->second;
}

dbNonDefaultRule*
dbDesign::getNonDefaultRuleByName(const std::string& name)
{ 
  auto itr = str2dbNonDefaultRule_.find(name);
  
  if(itr == str2dbNonDefaultRule_.end())  
    return nullptr;
  else
    return itr->second;
}

dbTrackGrid*
dbDesign::getTrackGridByLayer(dbLayer* layer)
{
  auto itr = layer2TrackGrid_.find(layer);
  
  if(itr == layer2TrackGrid_.end())  
    return nullptr;
  else
    return itr->second;
}

void
dbDesign::setDbu(int dbu)
{
  int lefDbu = tech_->getDbu();
  int defDbu = dbu;

  if(lefDbu != defDbu)
  {
    printf("DEF Dbu (%d) is different from LEF Dbu (%d)\n", 
            defDbu, lefDbu);
    exit(1);
  }
}

void
dbDesign::setDivider(const char div)
{
  const char lefDiv = tech_->getDivider();
  const char defDiv = div;

  if(lefDiv != defDiv)
  {
    printf("DEF Div (%c) is different from LEF Div (%c)\n", 
            defDiv, lefDiv);
    exit(1);
  }
}

void
dbDesign::setDie(const defiBox* box)
{
  defiPoints pts = box->getPoint();
  int numPoints = pts.numPoints;

  if(numPoints != 2)
  {
    printf("Does not support non-rectangular die...\n");
    exit(1);
  }

  int lx = box->xl();
  int ly = box->yl();
  int ux = box->xh();
  int uy = box->yh();

  die_ = new dbDie;

  die_->setLx(lx);
  die_->setLy(ly);
  die_->setUx(ux);
  die_->setUy(uy);

  //die_.print();
}

void
dbDesign::addNewRow(const defiRow* ro)
{
  dbRow* newRow = new dbRow;

  newRow->setName( std::string( ro->name() ) );

  dbSite* site = tech_->getSiteByName( std::string(ro->macro()) );

  if(site == nullptr) // macro() returns site name??
    techNotExist("Site", ro->macro());
  
  newRow->setSite(site);
  newRow->setOrigX( static_cast<int>(ro->x()) );
  newRow->setOrigY( static_cast<int>(ro->y()) );

  auto ori = types_->getOrient( std::string(ro->orientStr()) );
  newRow->setOrient( ori );

  if( ro->hasDo() )
  {
    newRow->setNumSiteX( ro->xNum() );
    newRow->setNumSiteY( ro->yNum() );
  }

  if( ro->hasDoStep() ) 
  {
    int stepX = ro->xStep();
    int stepY = ro->yStep();

    if(stepX > 1 && stepX < site->sizeX())
    {
      printf("Row StepX %d is smaller than site width %d\n", stepX, site->sizeX());
      exit(1);
    }

    if(stepY > 1 && stepY < site->sizeY())
    {
      printf("Row StepY %d is smaller than site height %d\n", stepY, site->sizeY());
      exit(1);
    }
    newRow->setStepX( stepX );
    newRow->setStepY( stepY );
  }

  rows_.push_back(newRow);
  //newRow->print();
  
  if(coreLx_ > newRow->lx()) coreLx_ = newRow->lx();
  if(coreLy_ > newRow->ly()) coreLy_ = newRow->ly();
  if(coreUx_ < newRow->ux()) coreUx_ = newRow->ux();
  if(coreUy_ < newRow->uy()) coreUy_ = newRow->uy();
}

void
dbDesign::addNewInst(const defiComponent* comp, const std::string& name)
{
  dbInst* newInst = new dbInst;
  insts_.push_back( newInst );

  // This is really weird part of LEF/DEF C++ API.
  // id() of defiComponent returns the name of the instance,
  // and name() returns the name of the LEF MACRO.
  // macroName() even returns null pointer...
  newInst->setName( name );

  if(duplicateCheck(str2dbInst_, name))
    alreadyExist("Inst", name.c_str());

  str2dbInst_[name] = newInst;
  
  dbMacro* macro = tech_->getMacroByName( std::string(comp->name()) );

  if(macro == nullptr)
    techNotExist("Macro", comp->name());

  newInst->setMacro( macro );

  fillInst(comp, newInst);

  // dbITerms are created at the same time with dbInst
  const std::string divStr = std::string(1, divider_); 
  // convert a single char to std::string
  for(auto mterm : macro->getMTerms())
  {
    const std::string itermName 
      = name + divStr + mterm->name();
    dbITerm* newITerm = new dbITerm(itermName, newInst, mterm);
    iterms_.push_back(newITerm);
    newInst->addITerm(newITerm);
  }
}

void
dbDesign::fillInst(const defiComponent* comp, dbInst* inst)
{ 
  auto orient = types_->getOrient( std::string( comp->placementOrientStr() ) );
  inst->setOrient( orient );

  auto status = types_->getStatus( comp->placementStatus() );
  inst->setStatus( status );

  inst->setLocation(comp->placementX(), comp->placementY());

  if(comp->hasSource())
  {
    auto source = types_->getSource( std::string( comp->source() ) );
    inst->setSource( source );
  }

  if(comp->hasHalo() > 0) 
  {
    int left, bottom, right, top;
    // haloEdges is non-const method originally in LEF/DEF C++ APIs,
    // so we have to change haloEdges to const method.
    comp->haloEdges(&left, &bottom, &right, &top);
    inst->setHalo(top, bottom, left, right);
  }

  if(comp->hasRouteHalo() > 0) 
    unsupportSyntax("ROUTEHALO");

  //inst->print();
}

void 
dbDesign::addNewIO(const defiPin* pin, const std::string& name)
{
  dbBTerm* newBTerm = new dbBTerm;
  bterms_.push_back( newBTerm );
  newBTerm->setName( name );

  if(duplicateCheck(str2dbBTerm_, name))
    alreadyExist("IO Pin", name.c_str());

  str2dbBTerm_[name] = newBTerm;
  
  const std::string netNameStr = pin->netName();
  dbNet* net = getNetByName( netNameStr );

  if(net == nullptr)
    net = getNewNet( netNameStr );

  newBTerm->setNet(net);

  if(pin->hasDirection())
  {
    auto dir = types_->getPinDirection( std::string(pin->direction()) );
    newBTerm->setDirection( dir );
  }

  // Pin has multiple ports
  if(pin->hasPort())
  {
    defiPinPort* port;
    int numPort = pin->numPorts(); 

    for(int portIdx = 0; portIdx < numPort; portIdx++)
    {
      port = pin->pinPort(portIdx);

      int origX = 0;
      int origY = 0;
      Orient orient = Orient::N; 
      Status status = Status::UNPLACED;
      // Initialized by default value

      if(port->hasPlacement())
      {
        origX = port->placementX();
        origY = port->placementY();
        orient = types_->getOrient( port->orientStr() );

        if(port->isPlaced())
          status = Status::PLACED;
        else if(port->isCover())
          status = Status::COVER;
        else if(port->isFixed())
          status = Status::FIXED;
        else
          status = Status::UNPLACED;
      }

      dbBTermPort* newBTermPort = new dbBTermPort;
      newBTermPort->setOrigX( origX );
      newBTermPort->setOrigY( origY );
      newBTermPort->setOrient( orient );
      newBTermPort->setStatus( status );

      int numLayerPort = port->numLayer();
      for(int layerIdx = 0; layerIdx < numLayerPort; layerIdx++)
      {
        // No support for LAYER MASK syntax.
        assert( port->layerMask(layerIdx) == 0 ); 

        int xl, yl, xh, yh;
        port->bounds(layerIdx, &xl, &yl, &xh, &yh);
        dbLayer* layer = tech_->getLayerByName( std::string(port->layer(layerIdx)) );

        if(layer == nullptr)
          techNotExist("Layer", pin->layer(layerIdx));

        newBTermPort->setOffsetLx(xl);
        newBTermPort->setOffsetLy(yl);
        newBTermPort->setOffsetUx(xh);
        newBTermPort->setOffsetUy(yh);
        newBTermPort->setLayer(layer);
        newBTermPort->setLocation();
        newBTerm->addPort( newBTermPort );
      }
    }
  }
  else // Pin has only one port
  {
    int origX = 0;
    int origY = 0;
    Orient orient = Orient::N; 
    Status status = Status::UNPLACED;
    // Initialized by default value

    if(pin->hasPlacement())
    {
      origX = pin->placementX();
      origY = pin->placementY();
      orient = types_->getOrient( pin->orientStr() );

      if(pin->isPlaced())
        status = Status::PLACED;
      else if(pin->isCover())
        status = Status::COVER;
      else if(pin->isFixed())
        status = Status::FIXED;
      else
        status = Status::UNPLACED;
    }
  
    if(pin->hasLayer())
    {
      dbBTermPort* newBTermPort = new dbBTermPort;
      newBTermPort->setOrigX( origX );
      newBTermPort->setOrigY( origY );
      newBTermPort->setOrient( orient );
      newBTermPort->setStatus( status );

      for(int i = 0; i < pin->numLayer(); i++)
      {
        int xl, yl, xh, yh;
        pin->bounds(i, &xl, &yl, &xh, &yh);
        dbLayer* layer = tech_->getLayerByName( std::string(pin->layer(i)) );

        if(layer == nullptr)
          techNotExist("Layer", pin->layer(i));

        newBTermPort->setOffsetLx(xl);
        newBTermPort->setOffsetLy(yl);
        newBTermPort->setOffsetUx(xh);
        newBTermPort->setOffsetUy(yh);
        newBTermPort->setLayer(layer);
        newBTermPort->setLocation();
        newBTerm->addPort( newBTermPort );
        // printf("(%d %d) (%d %d)\n", xl, yl, xh, yh);
      }
    }
  }

  // newBTerm->print();
}

dbNet*
dbDesign::getNewNet(const std::string& name)
{
  dbNet* newNet = new dbNet;
  nets_.push_back( newNet );
  newNet->setName(name);

  if(duplicateCheck(str2dbNet_, name))
    alreadyExist("Net", name.c_str());

  str2dbNet_[name] = newNet;

  return newNet;
}

void
dbDesign::fillNet(const defiNet* defNet, dbNet* net)
{
  if(defNet->hasUse())    
  {
    auto use = types_->getNetUse(std::string(defNet->use()));
    net->setUse(use);
  }

  if(defNet->hasSource()) 
  {
    auto src = types_->getSource(std::string(defNet->source()));
    net->setSource(src);
  }

  if(defNet->hasNonDefaultRule())
  {
    auto dbNdr 
      = getNonDefaultRuleByName(std::string(defNet->nonDefaultRule()));
    net->setNonDefaultRule(dbNdr);
  }

  for(int i = 0; i < defNet->numConnections(); ++i) 
  {
    if(defNet->pinIsSynthesized(i)) 
      unsupportSyntax("SYNTHESIZED");

    if(defNet->pinIsMustJoin(i)) 
      unsupportSyntax("MUSTJOIN");
    else 
    {
      const std::string& instNameStr 
        = removeBackSlashBracket( std::string(defNet->instance(i)) );
      const std::string& termNameStr = std::string(defNet->pin(i));

      if(instNameStr == "PIN" || instNameStr == "Pin" || instNameStr == "pin")
      {
        dbBTerm* bterm = getBTermByName(termNameStr);
        assert(bterm != nullptr);
        net->addBTerm(bterm);
        bterm->setNet(net);
      }
      else if(net->isSpecial() == false)
      {
        dbInst* inst = getInstByName(instNameStr);
        assert(inst != nullptr);

        dbITerm* iterm = inst->getITermByMTermName(termNameStr);
        assert(iterm != nullptr);
        
        net->addITerm(iterm);
        iterm->setNet(net);
      }
    }
  }
 
  const int numWires = defNet->numWires();
  for(int wID = 0; wID < numWires; wID++)
  {
    dbWire* newWire = net->getWire();
    const defiWire* defWire = defNet->wire(wID);

    const int numPaths = defWire->numPaths();
    for(int pID = 0; pID < numPaths; pID++)
    {
      const defiPath* defPath = defWire->path(pID);
      defPath->initTraverse();

      // Path Information
      std::vector<std::pair<int, int>> points; // 1. Points (size must be one or two)
      bool hasNDR = net->hasNonDefaultRule();  // 2. Check if NDR is applied (true if TAPER RULE, false if TAPER).
      bool isPatchWire = false;           // 3. Check if RECT keyword is on. 
                                          //    (I don't know why but Innovus calls this as a 'patch wire')
      dbLayer* layer = nullptr;           // 4. Tell us the bottom layer of this path.
      dbViaMaster* viaMaster = nullptr;   // 5. This will be valid if the path has a via.
      int pathX, pathY, pathExt;          // 6. The path coordinate. 
      int rectLx, rectLy, rectUx, rectUy; // 7. Coordinates of RECT.
      int wireMask = 0;                   // 8. Wire Mask (will be zero if none)
      int viaTopMask = 0;                 // 9. Via Mask (will be zero if none)
      int viaCutMask = 0;
      int viaBotMask = 0;

      int pathId;
      while((pathId = defPath->next()) != DEFIPATH_DONE) 
      {
        switch(pathId) 
        {
          case DEFIPATH_LAYER: 
          {
            const char* layerName = defPath->getLayer();
            layer = tech_->getLayerByName(std::string(layerName));

            if(layer == nullptr)
              techNotExist("Layer", layerName);

            int nextId = defPath->next();
            if(nextId == DEFIPATH_TAPER)
            {
              // TAPER keyword makes the next wire segment
              // is not affected by NonDefaultRule.
              hasNDR = false;
              // unsupportSyntax("TAPER");
            }
            else if(nextId == DEFIPATH_TAPERRULE)
            {
              // TAPER RULE keyword makes the next wire segment
              // is affected by the following NonDefaultRule.
              unsupportSyntax("TAPER RULE");
            }
            else
              defPath->prev();
            break;
          }
          case DEFIPATH_VIA: 
          {
            const char* viaName = defPath->getVia();

            viaMaster = tech_->getViaMasterByName(std::string(viaName));
            if(viaMaster == nullptr)
            {
              viaMaster = this->getViaMasterByName(std::string(viaName));
              if(viaMaster == nullptr)
                designNotExist("Via", viaName);
            }

            int nextId = defPath->next();
            if(nextId == DEFIPATH_VIAROTATION) 
              unsupportSyntax("VIA ROTATION");
            else 
              defPath->prev();  
            break;
          }

          case DEFIPATH_POINT: 
          {
            defPath->getPoint(&pathX, &pathY);
            points.push_back( {pathX, pathY} );
            break;
          }

          // NOTE : We assume FLUSHPOINT is not called for via.
          case DEFIPATH_FLUSHPOINT:
          {
            // According to LEF DEF REF,
            // ext specifies the amount by which the wire 
            // is extended past the endpoint of the segment.
            defPath->getFlushPoint(&pathX, &pathY, &pathExt);
            points.push_back( {pathX, pathY} );
            if(pathExt != 0)
              unsupportSyntax("EXT VALUE");
            break;
          }

          case DEFIPATH_MASK:
          {
            // According to LEF DEF REF,
            // MASK maskNum specifies which mask for 
            // double or triple patterning lithography to
            // use for the next wire or RECT.
            // (sometimes maskNum is referred to as 'color'.
            wireMask = defPath->getMask();
            break;
          }

          case DEFIPATH_VIAMASK:
          {
            viaBotMask = defPath->getViaBottomMask();
            viaCutMask = defPath->getViaCutMask();
            viaTopMask = defPath->getViaTopMask();
            break;
          }

          case DEFIPATH_RECT: 
          {
            defPath->getViaRect(&rectLx, &rectLy, &rectUx, &rectUy);
            break;
          }

          case DEFIPATH_STYLE:
          {
            unsupportSyntax("STYLE");
            break;
          }

          case DEFIPATH_VIRTUALPOINT:
          {
            // unsupportSyntax("VIRTUAL");
            break;
          }

          default:
            break;
        }

      }

      int numPathPoints = points.size();

      assert(numPathPoints == 1 || numPathPoints == 2);

      int pathX1 = points[0].first;
      int pathY1 = points[0].second;
      
      dbWireSegment* newSeg = new dbWireSegment;
      newWire->addWireSegment(newSeg);

      // chracterizing the new segment.
      newSeg->setLayer(layer);
      newSeg->setStartXY(pathX1, pathY1);
      newSeg->setMask(wireMask);

      int width = 0;
      if(hasNDR)
      {
        auto ndr = net->getNonDefaultRule();
        auto layer_rule = ndr->getLayerRuleByName(layer->name());
        newSeg->setRule(layer_rule);
        assert(layer_rule != nullptr);
        width = layer_rule->width;
      }
      else
      {
        width = layer->width();
      }

      if(numPathPoints == 2)
      {
        int pathX2 = points[1].first;
        int pathY2 = points[1].second;

        newSeg->setEndXY(pathX2, pathY2);
      }
      else
      {
        if(isPatchWire) // does not use spec of default layer
        {
          newSeg->setPatch();
          newSeg->setStartXY(rectLx, rectLy);
          newSeg->setEndXY(rectUx, rectUy);
        }
        else
        {
          newSeg->setStartXY(pathX1, pathY1);
          newSeg->setEndXY(pathX1, pathY1);
        }
      }

      // If via exists
      if(viaMaster != nullptr)
      {
        // TODO : Add Via Size / Coordinates
        dbViaInst* newViaInst = new dbViaInst;
        newViaInst->setMaster(viaMaster);
        newViaInst->setBotMask(viaBotMask);
        newViaInst->setCutMask(viaCutMask);
        newViaInst->setTopMask(viaTopMask);
        newSeg->setVia(newViaInst);
      }
    }
  }
  // net->print();
}

void
dbDesign::addNewViaMaster(const defiVia* defVia)
{
  dbGeneratedVia* newVia = new dbGeneratedVia;
  dbViaMaster* newMaster = new dbViaMaster(newVia);

  vias_.push_back(newMaster);
  newVia->setName(defVia->name());
  str2dbViaMaster_[newVia->name()] = newMaster;

  if(defVia->hasViaRule())
  {
    char* viaRuleName;
    int xSize ,ySize;
    char* botLayerName;
    char* cutLayerName;
    char* topLayerName;
    int xCutSpacing, yCutSpacing;
    int xBotEnc, yBotEnc, xTopEnc, yTopEnc;
    defVia->viaRule(&viaRuleName,
                    &xSize,
                    &ySize,
                    &botLayerName,
                    &cutLayerName,
                    &topLayerName,
                    &xCutSpacing,
                    &yCutSpacing,
                    &xBotEnc,
                    &yBotEnc,
                    &xTopEnc,
                    &yTopEnc);

    dbLayer* botLayer = tech_->getLayerByName( botLayerName );
    dbLayer* cutLayer = tech_->getLayerByName( cutLayerName );
    dbLayer* topLayer = tech_->getLayerByName( topLayerName );

    newVia->setCutSizeX( xSize );
    newVia->setCutSizeY( ySize );

    newVia->setBotLayer( botLayer );
    newVia->setCutLayer( cutLayer );
    newVia->setTopLayer( topLayer );

    newVia->setCutSpacingX( xCutSpacing );
    newVia->setCutSpacingY( yCutSpacing );

    newVia->setEnclosure(xBotEnc, yBotEnc, xTopEnc, yTopEnc);
  }

  if(defVia->hasRowCol()) 
  {
    int numCutRows;
    int numCutCols;
    defVia->rowCol(&numCutRows, &numCutCols);
    newVia->setNumRow(numCutRows);
    newVia->setNumCol(numCutCols);
  }
}

void
dbDesign::addNewNonDefaultRule(const defiNonDefault* defNdr)
{
  dbNonDefaultRule* newNdr = new dbNonDefaultRule;
  newNdr->setName( std::string(defNdr->name()) );

  ndrs_.push_back(newNdr);
  str2dbNonDefaultRule_[newNdr->name()] = newNdr;

  if(defNdr->hasHardspacing())
    newNdr->setHardSpacing();

  const int numLayer = defNdr->numLayers();

  for(int i = 0; i < numLayer; i++) 
  {
    dbLayer* layerForThisRule 
      = tech_->getLayerByName(std::string(defNdr->layerName(i)));

    int layerSpacing = 0;
    int layerWidth = defNdr->layerWidthVal(i);

    if(defNdr->hasLayerSpacing(i)) 
      layerSpacing = defNdr->layerSpacingVal(i);

    if(defNdr->hasLayerDiagWidth(i)) 
      unsupportSyntax("DIAGWIDTH");

    if(defNdr->hasLayerWireExt(i)) 
      unsupportSyntax("WIRE EXT");

    dbLayerRule* newRule = new dbLayerRule;
    newRule->layer = layerForThisRule;
    newRule->width = layerWidth;
    newRule->spacing = layerSpacing;

    newNdr->addLayerRule( newRule );
  }

  const int numVia = defNdr->numVias();
  for(int i = 0; i < numVia; i++) 
  {
    std::string viaName = std::string(defNdr->viaName(i));
    dbViaMaster* viaMaster = tech_->getViaMasterByName(viaName);

    if(viaMaster == nullptr)
      viaMaster = this->getViaMasterByName(viaName);

    if(viaMaster == nullptr)
      techNotExist("Via", viaName.c_str());

    newNdr->addVia(viaMaster);
  }

  const int numViaRule = defNdr->numViaRules();
  if(numViaRule > 0)
    unsupportSyntax("DEF VIARULE");

  const int numMinCut = defNdr->numMinCuts();
  if(numMinCut > 0)
    unsupportSyntax("DEF MINCUT");
}

void
dbDesign::addNewBlockage(const defiBlockage* defBlk)
{
  dbBlockage* newBlk = new dbBlockage;
  blockages_.push_back(newBlk);
  
  if(defBlk->hasLayer())
  {
    dbLayer* layer 
      = tech_->getLayerByName(std::string(defBlk->layerName()));

    if(layer == nullptr)
      techNotExist("Layer", defBlk->layerName());

    newBlk->setLayer(layer);
  }
  else 
    newBlk->setPlacementBlockage();

  if(defBlk->hasSpacing())
    newBlk->setSpacing(defBlk->minSpacing());

  const int numRect = defBlk->numRectangles();
  if(numRect > 1)
    unsupportSyntax("Multiple RECT BLOCKAGE");
  else if(numRect == 1)
  {
    int lx = defBlk->xl(0);
    int ly = defBlk->yl(0);
    int ux = defBlk->xh(0);
    int uy = defBlk->yh(0);
    newBlk->addPoint(lx, ly);
    newBlk->addPoint(ux, uy);
    newBlk->updateBBox();
    return;
  }
  else
  {
    const int numPoly = defBlk->numPolygons();
    assert(numPoly == 1);
    defiPoints polyPts = defBlk->getPolygon(0);

    const int numPts = polyPts.numPoints;
    for(int j = 0; j < numPts; j++)
      newBlk->addPoint(polyPts.x[j], polyPts.y[j]);
    newBlk->updateBBox();
  }
}

void
dbDesign::addNewTrack(const defiTrack* defTrack)
{
  int firstTrackMask = defTrack->firstTrackMask();
  int sameMask = defTrack->sameMask();

  const int numLayer = defTrack->numLayers();
  assert(numLayer == 1);

  const char direction = defTrack->macro()[0];

  // I don't know why, but DO is stored as xNum
  const int numDo   = defTrack->xNum(); 
  const int start   = defTrack->x();
  const int spacing = defTrack->xStep(); // STEP in DEF

  dbLayer* layer 
    = tech_->getLayerByName(std::string(defTrack->layer(0)));

  dbTrackGrid* trackGrid;

  if(layer == nullptr)
    techNotExist("Layer,", defTrack->layer(0));
  else
  {
    trackGrid = getTrackGridByLayer(layer);
    if(trackGrid == nullptr)
    {
      trackGrid = new dbTrackGrid;
      trackGrid->setLayer(layer);
      layer2TrackGrid_[layer] = trackGrid;
      trackGrids_.push_back(trackGrid);
    }
  }

  for(int i = 0; i < numDo; i++)
  {
    int newStart = start + i * spacing;
    dbTrack newTrack = {newStart, spacing};
    // order of constructor parameter matters!

    if(direction == 'X')
      trackGrid->addVTrack(newTrack);
    else
      trackGrid->addHTrack(newTrack);
  }
}

void
dbDesign::addNewSNet(const defiNet* defsnet)
{
  dbNet* newNet = new dbNet;
  
  std::string name = std::string(defsnet->name());

  snets_.push_back( newNet );
  newNet->setName(name);
  newNet->setSpecial();

  if(duplicateCheck(str2dbNet_, name))
    alreadyExist("Special Net", name.c_str());

  str2dbNet_[name] = newNet;

  fillNet(defsnet, newNet);
}

void
dbDesign::writeDef(const char* path) const
{
  //const std::filesystem::path filename_with_path(std::string(path));
  const int dbu = tech_->getDbu();

  // Step #1. Make Def file
  std::ofstream def_output;
  def_output.open(std::string(path));

  def_output << "# Created by SkyPlace (jkim97@postech.ac.kr)\n";
  def_output << "# " << getCalenderDate() << " " << getClockTime() << "\n";
  def_output << "VERSION 5.8 ;\n";
  def_output << "DIVIDERCHAR \"/\" ;\n";
  def_output << "BUSBITCHARS \"[]\" ;\n";
  def_output << "DESIGN " << name_ << " ;\n";
  def_output << "UNITS DISTANCE MICRONS " << dbu << " ;\n";

  // Step #2. Write DIEAREA Section
  int dieLx = die_->lx();
  int dieLy = die_->ly();
  int dieUx = die_->ux();
  int dieUy = die_->uy();

  def_output << "DIEAREA ";
  def_output << "( " << dieLx << " " << dieLy << " ) ";
  def_output << "( " << dieUx << " " << dieUy << " ) ;\n";

  // Step #3. Write ROW Section
  for(const auto& row : rows_)
  {
    def_output << "ROW " << row->name() << " " << row->site()->name() << " ";
    def_output << row->origX() << " " << row->origY() << " ";

    auto orient = row->orient();
    if(orient == Orient::N)
      def_output << "N ";
    else if(orient == Orient::FS)
      def_output << "FS ";
    else 
      def_output << "N ";
    // We only support N and FS for dbRow yet

    def_output << "DO "   << row->numSiteX() << " BY " << row->numSiteY() << " ";
    def_output << "STEP " << row->stepX() << " " << row->stepY() << " ;\n";
  }

  // Step #4. Write COMPONENTS Section
  def_output << "\n";
  def_output << "COMPONENTS " << insts_.size() << " ;\n";

  for(auto& inst : insts_)
  {
    def_output << "    - ";
    def_output << inst->name() << " ";
    def_output << inst->macro()->name() << " ";

    def_output << "+ SOURCE ";
    const auto source = inst->source();
    if(source == Source::DIST)
      def_output << "DIST ";
    else if(source == Source::NETLIST)
      def_output << "NETLIST ";
    else if(source == Source::TIMING)
      def_output << "TIMING ";
    else if(source == Source::USER)
      def_output << "USER ";

    def_output << "+";
    if(inst->isFixed())
      def_output << " FIXED ";
    else
      def_output << " PLACED ";

    def_output << "( " << inst->lx() << " " << inst->ly() << " )";

    const auto orient = inst->orient();
    if(orient == Orient::N)
      def_output << " N ;\n";
    else if(orient == Orient::S)
      def_output << " S ;\n";
    else if(orient == Orient::FN)
      def_output << " FN ;\n";
    else if(orient == Orient::FS)
      def_output << " FS ;\n";
  }

  def_output << "END COMPONENTS" << std::endl;

  // Step #5. Write PINS Section
  def_output << "\n";
  def_output << "PINS " << bterms_.size() << " ;\n";

  for(const auto& bterm : bterms_)
  {
    def_output << "    - ";
    def_output << bterm->name() << " + NET " << bterm->net()->name();
    def_output << " + DIRECTION ";

    const auto direction = bterm->direction();
    if(direction == PinDirection::INPUT)
      def_output << "INPUT\n";
    else if(direction == PinDirection::OUTPUT)
      def_output << "OUTPUT\n";
    else if(direction == PinDirection::INOUT)
      def_output << "INOUT\n";

    def_output << "      + PORT" << std::endl;

    for(const auto& port : bterm->ports())
    {
      int offsetLx = port->offsetLx();
      int offsetLy = port->offsetLy();
      int offsetUx = port->offsetUx();
      int offsetUy = port->offsetUy();
      def_output << "        + LAYER " << port->layer()->name() << " ";
      def_output << " ( " << offsetLx << " " << offsetLy << " )";
      def_output << " ( " << offsetUx << " " << offsetUy << " )";

      def_output << std::endl;
      const auto status = port->status();
      if(status == Status::PLACED)
        def_output << "        + PLACED ";
      else if(status == Status::FIXED)
        def_output << "        + FIXED ";
      else if(status == Status::UNPLACED)
        def_output << "        + UNPLACED ";
      else if(status == Status::COVER)
        def_output << "        + COVER ";
      else
        def_output << "        + PLACED ";

      int origX = port->origX();
      int origY = port->origY();
      def_output << " ( " << origX << " " << origY << " ) ";

      const auto orient = port->orient();
      switch(orient)
      {
        case Orient::N :
        {
          def_output << "N ;";
          break;
        }
        case Orient::S :
        {
          def_output << "S ;";
          break;
        }
        case Orient::W :
        {
          def_output << "W ;";
          break;
        }
        case Orient::E :
        {
          def_output << "E ;";
          break;
        }
        case Orient::FN :
        {
          def_output << "FN ;";
          break;
        }
        case Orient::FS :
        {
          def_output << "FS ;";
          break;
        }
        case Orient::FW :
        {
          def_output << "FW ;";
          break;
        }
        case Orient::FE :
        {
          def_output << "FE ;";
          break;
        }
      }
    }
    def_output << std::endl;
  }

  def_output << "END PINS" << std::endl;

  // Step #6. Write NETS Section
  def_output << std::endl;
  def_output << "NETS " << nets_.size() << " ;\n";

  for(const auto& net : nets_)
  {
    std::string netName = net->name();
    //modifyNetName(net, netName);

    def_output << "    - " << netName << " ";

    for(const auto& bterm : net->getBTerms())
    {
      def_output << "( PIN ";
      def_output << bterm->name() << " ";
      def_output << ") ";
    }

    for(const auto& iterm : net->getITerms())
    {
      def_output << "( ";
      def_output << iterm->getInst()->name() << " ";
      def_output << iterm->getMTerm()->name() << " ";
      def_output << ") ";
    }

    def_output << "+ USE ";
    const auto usage = net->use();
    switch(usage)
    {
      case NetUse::SIGNAL_NET :
      {
        def_output << "SIGNAL";
        break;
      }
      case NetUse::CLOCK_NET:
      {
        def_output << "CLOCK";
        break;
      }
      case NetUse::POWER_NET:
      {
        def_output << "POWER";
        break;
      }
      case NetUse::GROUND_NET:
      {
        def_output << "GROUND";
        break;
      }
      case NetUse::RESET_NET:
      {
        def_output << "RESET";
        break;
      }
      default:
        assert(0);
    }

    const auto wire = net->getWire();
    const auto& segments = wire->getSegments();
    const auto numSeg = segments.size();

    for(const auto& seg : segments)
    {
      def_output << std::endl;
      def_output << "      ";
      if(seg == segments.front())
        def_output << "+ ROUTED ";
      else
        def_output << "NEW ";

      def_output << seg->layer()->name() << " ";
      
      int x1 = seg->startX();
      int y1 = seg->startY();

      def_output << "( " << x1 << " " << y1 << " ) ";

      if(seg->hasVia() == true)
        def_output << seg->getVia()->getMaster()->name();
      else
      {
        int x2 = seg->endX();
        int y2 = seg->endY();

        if(x1 == x2)
          def_output << "( * " << y2 << " ) ";
        else if(y1 == y2)
          def_output << "( " << x2 << " * ) ";
        else
          assert(0);
      }

    }

    def_output << " ;" << std::endl;
  }

  def_output << "END NETS" << std::endl;

  // Step #7. Write SPECIAL NETS
  def_output << "SPECIALNETS " << snets_.size() << " ;\n";

  for(const auto& net : snets_)
  {
    std::string netName = net->name();
    def_output << "    - " << netName << " ";
    def_output << "( * " << netName << " ) ";
    def_output << "+ USE ";
    const auto usage = net->use();
    switch(usage)
    {
      case NetUse::SIGNAL_NET :
      {
        def_output << "SIGNAL";
        break;
      }
      case NetUse::CLOCK_NET:
      {
        def_output << "CLOCK";
        break;
      }
      case NetUse::POWER_NET:
      {
        def_output << "POWER";
        break;
      }
      case NetUse::GROUND_NET:
      {
        def_output << "GROUND";
        break;
      }
      case NetUse::RESET_NET:
      {
        def_output << "RESET";
        break;
      }
      default:
        assert(0);
    }

    const auto wire = net->getWire();
    const auto& segments = wire->getSegments();
    const auto numSeg = segments.size();

    for(const auto& seg : segments)
    {
      def_output << std::endl;
      def_output << "      ";
      if(seg == segments.front())
        def_output << "+ ROUTED ";
      else
        def_output << "NEW ";

      def_output << seg->layer()->name() << " ";
      
      int x1 = seg->startX();
      int y1 = seg->startY();

      def_output << "( " << x1 << " " << y1 << " ) ";

      if(seg->hasVia() == true)
        def_output << seg->getVia()->getMaster()->name();
      else
      {
        int x2 = seg->endX();
        int y2 = seg->endY();
        def_output << "( " << x2 << " " << y2 << " ) ";
      }
    }

    def_output << " ;" << std::endl;
  }

  def_output << "END SPECIALNETS" << std::endl;
  def_output << std::endl;

  def_output << "END DESIGN" << std::endl;
  def_output.close();

  printf("Write results to %s.\n", path);
}

void
dbDesign::writeBookShelf(const char* path) const
{
  const int dbu = tech_->getDbu();

  std::string auxFileName = std::string(path); 
  std::string auxFileNameWithOutSuffix = "";
  std::string auxFileNameWithOutSlash  = "";

  if(auxFileName == "")
  {
    auxFileName = name_ + ".aux";
    std::cout << "File name is not given... ";
    std::cout << "design name will be used by default..." << std::endl;
  }
  else
  {
    size_t dot   = auxFileName.find_last_of('.');
    size_t slash = auxFileName.find_last_of('/');

    std::string suffix = auxFileName.substr(dot + 1);
    auxFileNameWithOutSuffix = auxFileName.substr(0, dot);
    auxFileNameWithOutSlash  = auxFileName.substr(slash + 1, dot - slash - 1);

    if(suffix != "aux")
    {
      printf("BookShelf output should be .aux file!\n");
      exit(1);
    }
  }

  std::string plFileName = auxFileNameWithOutSuffix + ".pl";
  
  // Step #1. Write .aux file 
  std::ofstream aux_output;
  aux_output.open(auxFileName);

  aux_output << "RowBasedPlacement :" << " ";
  aux_output << name_ + ".nodes" << " ";
  aux_output << name_ + ".nets" << " ";
  aux_output << name_ + ".wts" << " ";
  // If wts filename is not written in aux, 
  // ntuplace3 binaray makes segmentation fault.
  aux_output << auxFileNameWithOutSlash + ".pl" << " ";
  aux_output << name_ + ".scl";

  aux_output.close();

  // Step #2. Write .pl file 
  // Print Headline
  std::ofstream pl_output;
  pl_output.open(plFileName);

  pl_output << "UCLA pl 1.0" << std::endl;
  pl_output << "# Created by SkyPlace (jkim97@postech.ac.kr)\n";
  pl_output << "# " << getCalenderDate() << " " << getClockTime() << "\n";
  pl_output << std::endl;

  for(auto& inst : insts_)
  {
    pl_output << inst->name() << " ";
    pl_output << static_cast<float>(inst->lx()) 
               / static_cast<float>(dbu) << " ";
    pl_output << static_cast<float>(inst->ly())
               / static_cast<float>(dbu) << " : N";
    if(inst->isFixed())
      pl_output << " /FIXED";
    pl_output << std::endl;
  }

  pl_output.close();
  printf("Write results to .aux (%s) and .pl (%s).\n", 
          auxFileName.c_str(), plFileName.c_str());
}

}
