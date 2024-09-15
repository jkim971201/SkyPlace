#include "GuiTrackGrid.h"

#include <iostream>
#include <limits>
#include <QStyleOptionGraphicsItem>

namespace gui
{

GuiTrackGrid::GuiTrackGrid(std::shared_ptr<GuiConfig> cfg,
                           const dbTrackGrid* grid)
  : grid_(grid)
{
  setConfig(cfg);

  const double dbu = static_cast<double>(getConfig()->dbu());

  const double dieLx = getConfig()->dieLx;
  const double dieLy = getConfig()->dieLy;
  const double dieUx = getConfig()->dieUx;
  const double dieUy = getConfig()->dieUy;

  // Vertical Tracks -> | | | | | ...
  for(const auto track : grid_->getVGrid())
  {
    double xOfThisTrack = static_cast<double>(track.start) / dbu;
    QPointF p1(xOfThisTrack, dieLy);
    QPointF p2(xOfThisTrack, dieUy);
    gui_tracks_.push_back(QLineF(p1, p2));
  }

  // Horizontal Trakcks
  for(const auto track : grid_->getHGrid())
  {
    double yOfThisTrack = track.start / dbu;
    QPointF p1(dieLx,yOfThisTrack);
    QPointF p2(dieUx,yOfThisTrack);
    gui_tracks_.push_back(QLineF(p1, p2));
  }

  rect_ 
    = QRectF(dieLx, dieLy, dieUx - dieLx, dieUy - dieLy);
}

QRectF
GuiTrackGrid::boundingRect() const
{
  return rect_;
}

void
GuiTrackGrid::paint(QPainter* painter, 
                    const QStyleOptionGraphicsItem* option,
                    QWidget* widget)
{
  const qreal lod 
    = option->levelOfDetailFromTransform(painter->worldTransform());

  if(lod < 20.0)
    return;

  const auto layer = grid_->layer();
  const auto color = getConfig()->getLayerColor(layer);

  auto& pen = getPen();

  pen.setWidthF(0);
  pen.setColor(color);

  painter->setPen(pen);

  for(const auto line : gui_tracks_)
    painter->drawLine(line);
}

}
