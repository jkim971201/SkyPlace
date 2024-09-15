#include <QStyleOptionGraphicsItem>
#include <iostream>

#include "GuiDie.h"

namespace gui
{

GuiDie::GuiDie(const dbDie* die)
  : die_ (die)
{}

QRectF
GuiDie::boundingRect() const
{
  return rect_;
}

void
GuiDie::paint(QPainter* painter, 
              const QStyleOptionGraphicsItem* option,
              QWidget* widget)
{
  // std::cout << "LoD : " << option->levelOfDetailFromTransform(painter->worldTransform()) << std::endl;
  painter->setPen( QPen(Qt::gray, 0, Qt::PenStyle::DashDotLine) );
  painter->drawRect(rect_);
}

}
