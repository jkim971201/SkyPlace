#include <cassert>
#include <limits>
#include <iostream>

#include <QApplication>
#include <QStyleOptionGraphicsItem>

#include "GuiPin.h"

namespace gui
{

GuiPin::GuiPin(std::shared_ptr<GuiConfig> cfg,
               bool isObs,
               const dbInst* inst,
               const dbLayer* layer,
               const std::vector<std::pair<int, int>>& shape)
  : isObs_  (isObs),
    inst_   (inst),
    layer_  (layer)
{
  setConfig(cfg);

  lx_ = std::numeric_limits<double>::min();
  ly_ = std::numeric_limits<double>::min();
  ux_ = std::numeric_limits<double>::max();
  uy_ = std::numeric_limits<double>::max();

  const Orient orient = inst_->orient();

  int originX_dbu;
  int originY_dbu;

  double originX_micron;
  double originY_micron;

  // Rotation Matrix
  // (x, y) -> (a1 * x + b1 * y, a2 * x + b2 * y)
  int a1 = 0;
  int b1 = 0;
  int a2 = 0;
  int b2 = 0;

  const double dbu = static_cast<double>(config_->dbu());

  switch(orient)
  {
    case Orient::N :
    {
      originX_dbu = inst_->lx();
      originY_dbu = inst_->ly();
      a1 = +1;
      b2 = +1;
      break;
    }
    case Orient::S :
    {
      originX_dbu = inst_->ux();
      originY_dbu = inst_->uy();
      a1 = -1;
      b2 = -1;
      break;
    }
    case Orient::W :
    {
      originX_dbu = inst_->ux();
      originY_dbu = inst_->ly();
      b1 = -1;
      a2 = +1;
      break;
    }
    case Orient::E :
    {
      originX_dbu = inst_->lx();
      originY_dbu = inst_->uy();
      b2 = +1;
      a2 = -1;
      break;
    }
    case Orient::FN :
    {
      originX_dbu = inst_->ux();
      originY_dbu = inst_->ly();
      a1 = -1;
      b2 = +1;
      break;
    }
    case Orient::FS :
    {
      originX_dbu = inst_->lx();
      originY_dbu = inst_->uy();
      a1 = +1;
      b2 = -1;
      break;
    }
    case Orient::FW :
    {
      originX_dbu = inst_->lx();
      originY_dbu = inst_->ly();
      b1 = +1;
      a2 = +1;
      break;
    }
    case Orient::FE :
    {
      originX_dbu = inst_->ux();
      originY_dbu = inst_->uy();
      b2 = -1;
      a2 = -1;
      break;
    }
    default:
      break;
  }

  originX_micron = originX_dbu / dbu;
  originY_micron = originY_dbu / dbu;

  lx_ = ux_ = originX_micron;
  ly_ = uy_ = originY_micron;

  for(const auto& [offsetX_dbu, offsetY_dbu] : shape)
  {
    double offsetX_micron = offsetX_dbu / dbu;
    double offsetY_micron = offsetY_dbu / dbu;
      
    double newX 
      = originX_micron + a1 * offsetX_micron + b1 * offsetY_micron;

    double newY 
      = originY_micron + a2 * offsetX_micron + b2 * offsetY_micron;

    polygon_.append(QPointF(newX, newY));

    if(lx_ > newX) lx_ = newX;
    if(ly_ > newY) ly_ = newY;
    if(ux_ < newX) ux_ = newX;
    if(uy_ < newY) uy_ = newY;
  }
}

QRectF
GuiPin::boundingRect() const 
{
  return QRectF(lx_, ly_, ux_ - lx_, uy_ - ly_);
}

void
GuiPin::paint(QPainter* painter, 
              const QStyleOptionGraphicsItem* option,
              QWidget* widget)
{
  QColor color = config_->getLayerColor(layer_);
  QBrush brush(color, Qt::BrushStyle::DiagCrossPattern);
  brush.setTransform(QTransform(painter->worldTransform().inverted()));

  painter->setPen( QPen(color, 0, Qt::PenStyle::SolidLine) );
  painter->setBrush( brush );
  painter->drawConvexPolygon(polygon_);
  // TODO : What if the PORT or OBS has non-convex shape?
}

}
