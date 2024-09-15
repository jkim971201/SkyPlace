#include <QApplication>
#include <QStyleOptionGraphicsItem>
#include <iostream>

#include "GuiBlockage.h"

namespace gui
{

GuiBlockage::GuiBlockage(std::shared_ptr<GuiConfig> cfg,
                         const dbBlockage* blk)
  : blk_(blk) 
{
  setConfig(cfg);

  const double dbu = static_cast<double>(getConfig()->dbu());

  int ioLx_dbu = blk_->lx();
  int ioLy_dbu = blk_->ly();
  int ioDx_dbu = blk_->dx();
  int ioDy_dbu = blk_->dy();

  double ioLx_micron = ioLx_dbu / dbu;
  double ioLy_micron = ioLy_dbu / dbu;
  double ioDx_micron = ioDx_dbu / dbu;
  double ioDy_micron = ioDy_dbu / dbu;

  rect_ = QRectF(ioLx_micron, ioLy_micron, ioDx_micron, ioDy_micron);
}

void
GuiBlockage::paint(QPainter* painter, 
                   const QStyleOptionGraphicsItem* option,
                   QWidget* widget)
{
  const auto layer = blk_->layer();
  const auto color = getConfig()->getLayerColor(layer);

  getPen().setColor(color);
  getBrush().setColor(color);
  getBrush().setStyle(Qt::BrushStyle::DiagCrossPattern);

  GuiRect::paint(painter, option, widget);
}

}
