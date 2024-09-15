#include <cassert>
#include <iostream>
#include <regex>

#include "db/dbTech.h"
#include "db/dbDesign.h"
#include "db/dbDie.h"
#include "db/dbInst.h"
#include "db/dbNet.h"
#include "db/dbITerm.h"
#include "db/dbBTerm.h"
#include "db/dbMTerm.h"
#include "db/dbRow.h"

#include "LayoutScene.h"
#include "gui_item/GuiDie.h"
#include "gui_item/GuiRow.h"
#include "gui_item/GuiInst.h"
#include "gui_item/GuiIO.h"
#include "gui_item/GuiPin.h"
#include "gui_item/GuiNet.h"
#include "gui_item/GuiBlockage.h"
#include "gui_item/GuiTrackGrid.h"

namespace gui
{

GuiConfig::GuiConfig(std::shared_ptr<dbTech> tech)
{
  dbu_ = tech->getDbu();
  QColor color;
  int routingLayerIdx = 0; // Index for ROUTING LAYER
  int cutLayerIdx     = 1; // Index for     CUT LAYER
  // Cut Layer matches to its upper layer.
  for(auto layer : tech->getLayers())
  {
    auto type = layer->type();
    if(type == RoutingType::ROUTING)
    {
      if(routingLayerIdx < QCOLOR_ARRAY.size())
      {
        color = QCOLOR_ARRAY.at(routingLayerIdx);
        routingLayerIdx++;
      }
      else 
        color = QColor(240, 255, 255); // color name : azure 
    }
    else if(type == RoutingType::CUT)
    {
      if(cutLayerIdx < QCOLOR_ARRAY.size())
      {
        color = QCOLOR_ARRAY.at(cutLayerIdx);
        cutLayerIdx++;
      }
      else 
        color = QColor(240, 255, 255); // color name : azure 
    }
    else
      color = QColor(240, 255, 255); // color name : azure 

    layer2Color_[layer] = color;
  }
}

const QColor
GuiConfig::getLayerColor(const dbLayer* layer)
{
  auto itr = layer2Color_.find(layer);
  if(itr == layer2Color_.end())
    assert(0); // TODO : Exception Handling

  return itr->second;
}

/*  LayoutScene   */
LayoutScene::LayoutScene(QObject* parent)
{
}

void
LayoutScene::setDatabase(std::shared_ptr<dbDatabase> db)
{
  db_ = db;
  config_ = std::make_shared<GuiConfig>(db_->getTech());

  // We keep die coordinates 
  // to share the boundary information of the layout
  // with all the gui items.
  const double dbu = static_cast<double>(config_->dbu());
  const auto die = db_->getDesign()->getDie();
  config_->dieLx = die->lx() / dbu;
  config_->dieLy = die->ly() / dbu;
  config_->dieUx = die->ux() / dbu;
  config_->dieUy = die->uy() / dbu;
}

void
LayoutScene::createGuiDie()
{
  const auto die = db_->getDesign()->getDie();
  GuiDie* die_gui = new GuiDie(die);

  double dieLx = config_->dieLx;
  double dieLy = config_->dieLy;
  double dieDx = config_->dieUx - dieLx;
  double dieDy = config_->dieUy - dieLy;

  die_gui->setRect( QRectF(dieLx, dieLy, dieDx, dieDy) );

  this->addItem(die_gui);
}

void
LayoutScene::createGuiRow()
{
  const double dbu = static_cast<double>(db_->getTech()->getDbu());

  for(auto row : db_->getDesign()->getRows())
  {
    GuiRow* row_gui = new GuiRow(row);

    double rowLx = static_cast<double>(row->lx()) / dbu;
    double rowLy = static_cast<double>(row->ly()) / dbu;
    double rowDx = static_cast<double>(row->dx()) / dbu;
    double rowDy = static_cast<double>(row->dy()) / dbu;
  
    row_gui->setRect( QRectF(rowLx, rowLy, rowDx, rowDy) );
    this->addItem(row_gui);
  }
}

void
LayoutScene::createGuiInst()
{
  const double dbu = static_cast<double>(db_->getTech()->getDbu());

  for(const auto inst : db_->getDesign()->getInsts())
  {
    GuiInst* inst_gui = new GuiInst(inst);
  
    double cellLx = static_cast<double>(inst->lx()) / dbu;
    double cellLy = static_cast<double>(inst->ly()) / dbu;
    double cellDx = static_cast<double>(inst->dx()) / dbu;
    double cellDy = static_cast<double>(inst->dy()) / dbu;
  
    inst_gui->setRect( QRectF(cellLx, cellLy, cellDx, cellDy) );
    this->addItem(inst_gui);

    // NOTE : Both PIN and OBS will be converted to GuiPin.
    for(const auto iterm : inst->getITerms())
    {
      const auto mterm = iterm->getMTerm();
      const auto macro = inst->macro();
  
      for(const auto port : mterm->ports())
      {
        // MASTERSLICE is a nonrouting layer.
        // LEF DEF REF (May 2017) Page 45
        const dbLayer* layer = port->layer();
        if(layer->type() == RoutingType::MASTERSLICE)
          continue;
  
        GuiPin* cell_pin = new GuiPin(config_, false, inst, layer, port->getShape());
        inst_gui->addGuiPin(cell_pin);
      }
  
      for(const auto obs : macro->getObs())
      {
        const dbLayer* layer = obs->layer();
        if(obs->layer()->type() == RoutingType::MASTERSLICE)
          continue;
  
        GuiPin* cell_obs = new GuiPin(config_, true , inst, layer, obs->getShape());
        inst_gui->addGuiPin(cell_obs);
      }
    }
  }
}

void
LayoutScene::createGuiIO()
{
  for(const auto bterm : db_->getDesign()->getBTerms())
  {
    GuiIO* io_gui  = new GuiIO(config_, bterm);
    this->addItem(io_gui);
  }
}

void
LayoutScene::createGuiNet()
{
  std::regex regBus("CH0_.*");
  for(const auto net : db_->getDesign()->getNets())
  {
    if(net->getWire()->getSegments().empty())
      continue;

    //if(!std::regex_match(net->name(), regBus))
    //  continue;
    
    GuiNet* net_gui  = new GuiNet(config_, net);
    this->addItem(net_gui);
  }
}

void
LayoutScene::createGuiBlockage()
{
  for(const auto blk : db_->getDesign()->getBlockages())
  {
    if(blk->isPlacementBlockage() == true)
      continue;

    GuiBlockage* blk_gui  = new GuiBlockage(config_, blk);
    this->addItem(blk_gui);
  }
}

void
LayoutScene::createGuiTrackGrid()
{
  for(const auto grid : db_->getDesign()->getTrackGrids())
  {
    if(grid->layer()->name() != "D12" &&
       grid->layer()->name() != "D11" &&
       grid->layer()->name() != "D10" &&
       grid->layer()->name() != "D9")
      continue;
    GuiTrackGrid* grid_gui  = new GuiTrackGrid(config_, grid);
    this->addItem(grid_gui);
  }
}

void
LayoutScene::expandScene()
{
  QRectF rect = sceneRect();
  double sceneW = rect.width();
  double sceneH = rect.height();

  // Make 10%  blank margin along boundary
  rect.adjust(-0.1 * sceneW, -0.1 * sceneH, 
              +0.1 * sceneW, +0.1 * sceneH);

  this->setSceneRect(rect);
}

}
