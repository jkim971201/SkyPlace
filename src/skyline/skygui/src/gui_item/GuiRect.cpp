#include <QApplication>
#include <QStyleOptionGraphicsItem>

#include "GuiRect.h"

namespace gui
{

GuiRect::GuiRect()
{
  rect_ = QRectF();
}

GuiRect::GuiRect(double lx, double ly, double dx, double dy)
{
  rect_ = QRectF(lx, ly, dx, dy);
}

QRectF
GuiRect::boundingRect() const
{
  return rect_;
}

void
GuiRect::paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option,
               QWidget* widget)
{
  const qreal lod 
    = option->levelOfDetailFromTransform(painter->worldTransform());

  brush_.setTransform(QTransform(painter->worldTransform().inverted()));
  // To avoid weird bug of Qt (rouding error issue?)
  // https://www.qtcentre.org/threads/2907-QBrush-pattern-problem

  pen_.setWidthF(0);
  pen_.setStyle(Qt::PenStyle::SolidLine);
  pen_.setJoinStyle(Qt::PenJoinStyle::MiterJoin);

  double instLineWidth = pen_.widthF() / lod;
  pen_.setWidthF(instLineWidth);

  painter->setPen(pen_);
  painter->setBrush(brush_);

  painter->drawLine(rect_.topLeft(), rect_.topRight());
  painter->drawLine(rect_.bottomLeft(), rect_.bottomRight());
  painter->drawLine(rect_.topLeft(), rect_.bottomLeft());
  painter->drawLine(rect_.topRight(), rect_.bottomRight());
  painter->drawRect(rect_);
}

}
