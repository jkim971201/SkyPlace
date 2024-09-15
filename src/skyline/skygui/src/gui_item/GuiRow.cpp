#include <QStyleOptionGraphicsItem>

#include "GuiRow.h"

namespace gui
{

GuiRow::GuiRow(dbRow* row)
  : row_ (row)
{

}

QRectF
GuiRow::boundingRect() const
{
  return rect_;
}

void
GuiRow::paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option,
               QWidget* widget)
{
  qreal lod = option->levelOfDetailFromTransform(painter->worldTransform());

  if(lod > 4.0)
  {
    painter->setPen(QPen(Qt::gray, 0, Qt::PenStyle::SolidLine));
    painter->drawRect(rect_);
  
    // Orientation Line
    painter->setPen(QPen(Qt::gray, 0, Qt::PenStyle::SolidLine));
    qreal row_width  = std::abs(rect_.width());
    qreal row_height = std::abs(rect_.height());
  
    qreal delta_x;
    qreal delta_y;
  
    if(row_width < row_height)
    {
      delta_x = row_width / 4.0;
      delta_y = row_width / 4.0;
    }
    else
    {
      delta_x = row_height / 4.0;
      delta_y = row_height / 4.0;
    }
  
    QPointF p1;
    QPointF p2;
  
    switch(row_->orient())
    {
      case Orient::N  :
      case Orient::FW :
      {
        QPointF tl = rect_.topLeft();
        p1 = tl + QPointF(0.0, delta_y);
        p2 = tl + QPointF(delta_x, 0.0);
        break;
      }
      case Orient::S  :
      case Orient::FE :
      {
        QPointF br = rect_.bottomRight();
        p1 = br + QPointF(0.0, -delta_y);
        p2 = br + QPointF(-delta_x, 0.0);
        break;
      }
      case Orient::W  :
      case Orient::FN :
      {
        QPointF tr = rect_.topRight();
        p1 = tr + QPointF(0.0, delta_y);
        p2 = tr + QPointF(-delta_x, 0.0);
        break;
      }
      case Orient::E  :
      case Orient::FS :
      {
        QPointF bl = rect_.bottomLeft();
        p1 = bl + QPointF(0.0, -delta_y);
        p2 = bl + QPointF(delta_x, 0.0);
        break;
      }
      default:
        break;
    }
  
    QLineF line(p1, p2);
    painter->drawLine(line);
  }
}

}
