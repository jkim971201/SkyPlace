#include <QApplication>
#include <QStyleOptionGraphicsItem>

#include <limits>
#include <iostream>
#include <cassert>

#include "GuiNet.h"

namespace gui
{

/* GuiWireSegment */
GuiWireSegment::GuiWireSegment(std::shared_ptr<GuiConfig> cfg,
                               const dbWireSegment* seg)
  : seg_(seg)
{
  setConfig(cfg);

  const double dbu = static_cast<double>(getConfig()->dbu());
  const double width = static_cast<double>(seg->width());

  int pathX1 = seg->startX();
  int pathY1 = seg->startY();
  int pathX2 = seg->endX();
  int pathY2 = seg->endY();

  double lx, ly, ux, uy;
  if(pathX1 == pathX2)
  {
    lx = static_cast<double>(pathX1 - width / 2);
    ux = static_cast<double>(pathX1 + width / 2);
    if(pathY1 < pathY2)
    {
      ly = static_cast<double>(pathY1 - width / 2);
      uy = static_cast<double>(pathY2 + width / 2);
    }
    else
    {
      ly = static_cast<double>(pathY2 - width / 2);
      uy = static_cast<double>(pathY1 + width / 2);
    }
  }
  else if(pathY1 == pathY2)
  {
    ly = static_cast<double>(pathY1 - width / 2);
    uy = static_cast<double>(pathY1 + width / 2);

    if(pathX1 < pathX2)
    {
      lx = static_cast<double>(pathX1 - width / 2);
      ux = static_cast<double>(pathX2 + width / 2);
    }
    else
    {
      lx = static_cast<double>(pathX2 - width / 2);
      ux = static_cast<double>(pathX1 + width / 2);
    }
  }
  else
    assert(1);

  double lx_micron = lx / dbu;
  double ly_micron = ly / dbu;
  double dx_micron = (ux - lx) / dbu;
  double dy_micron = (uy - ly) / dbu;

  rect_ = QRectF(lx_micron, ly_micron, dx_micron, dy_micron);
}

void
GuiWireSegment::paint(QPainter* painter, 
                      const QStyleOptionGraphicsItem* option,
                      QWidget* widget)
{
  const auto layer = seg_->layer();
  const auto color = getConfig()->getLayerColor(layer);

  getPen().setColor(color);
  getBrush().setColor(color);
  getBrush().setStyle(Qt::BrushStyle::DiagCrossPattern);

  GuiRect::paint(painter, option, widget);
}

/* GuiNet */
GuiNet::GuiNet(std::shared_ptr<GuiConfig> cfg,
               const dbNet* net)
  : net_(net) 
{
  setConfig(cfg);

  const double dbu = static_cast<double>(getConfig()->dbu());

  const auto wire = net->getWire();
  double lx_micron = std::numeric_limits<double>::max();
  double ly_micron = std::numeric_limits<double>::max();
  double ux_micron = std::numeric_limits<double>::min();
  double uy_micron = std::numeric_limits<double>::min();

  for(const auto wire_seg : wire->getSegments())
  { 
    GuiWireSegment* newSeg = new GuiWireSegment(getConfig(), wire_seg);
    gui_wires_.push_back(newSeg);

    const auto& segRect = newSeg->boundingRect();
    double segLx = segRect.left();
    double segLy = segRect.bottom();
    double segUx = segRect.right();
    double segUy = segRect.top();

    if(segLx < lx_micron) lx_micron = segLx;
    if(segLy < ly_micron) ly_micron = segLy;
    if(segUx > ux_micron) ux_micron = segUx;
    if(segUy > uy_micron) uy_micron = segUy;
  }

  double dx_micron = ux_micron - lx_micron;
  double dy_micron = uy_micron - ly_micron;

  rect_ = QRectF(lx_micron, ly_micron, dx_micron, dy_micron);
}

QRectF
GuiNet::boundingRect() const
{
  return rect_;
}

void
GuiNet::paint(QPainter* painter, 
              const QStyleOptionGraphicsItem* option,
              QWidget* widget)
{
  const qreal lod 
    = option->levelOfDetailFromTransform(painter->worldTransform());

  for(auto gui_wire : gui_wires_)
    gui_wire->paint(painter, option, widget);
}

}
