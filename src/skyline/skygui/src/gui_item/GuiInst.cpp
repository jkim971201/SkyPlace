#include <QApplication>
#include <QStyleOptionGraphicsItem>
#include <iostream>

#include "GuiInst.h"

namespace gui
{

GuiInst::GuiInst(const dbInst* inst)
  : inst_ (inst)
{
}

QRectF
GuiInst::boundingRect() const
{
  return rect_;
}

void
GuiInst::paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option,
               QWidget* widget)
{
  const qreal lod 
    = option->levelOfDetailFromTransform(painter->worldTransform());

  QPen pen(Qt::gray, 0, Qt::PenStyle::SolidLine);
  pen.setJoinStyle(Qt::PenJoinStyle::MiterJoin);

  const double instLineWidth = pen.widthF() / lod;
  pen.setWidthF(instLineWidth);
  painter->setPen(pen);

  // Inspired by iEDA
  if(lod < 10) 
  {
    painter->drawLine(rect_.topLeft(), rect_.topRight());
    painter->drawLine(rect_.bottomLeft(), rect_.bottomRight());
    painter->drawLine(rect_.topLeft(), rect_.bottomLeft());
    painter->drawLine(rect_.topRight(), rect_.bottomRight());
    return;
  }

  QBrush brush(Qt::gray, Qt::BrushStyle::Dense6Pattern);
  brush.setTransform(QTransform(painter->worldTransform().inverted()));
  painter->setBrush(brush);

  if(lod < 0.4) 
  {
    painter->drawRect(rect_);
    return;
  }

  painter->drawRect(rect_);

  // Orientation Line
  qreal inst_width  = std::abs(rect_.width());
  qreal inst_height = std::abs(rect_.height());

  qreal delta_x = inst_width  / 4.0;
  qreal delta_y = inst_height / 4.0;

  QPointF p1;
  QPointF p2;

  switch(inst_->orient())
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

  pen.setJoinStyle(Qt::PenJoinStyle::BevelJoin);
  painter->setPen(pen);
  painter->drawLine(line);

  //if(inst_->isMacro() || (!inst_->isMacro() && lod > 0.4) )
  //  drawInstName(painter, Qt::white, lod);

  if(lod > 20)
  {
    for(auto gui_pin : gui_pins_)
      gui_pin->paint(painter, option, widget);
  }
}

void
GuiInst::drawInstName(QPainter* painter, const QColor& color, qreal lod)
{
  const qreal rectW = rect_.width();
  const qreal rectH = rect_.height();

  const qreal scale_adjust   = 1.0 / lod;
  const qreal text_font_size = rectH * 0.05 * lod;

  const qreal rectW_scaled = rect_.width();
  const qreal rectH_scaled = rect_.height();

  QFont font = painter->font();
  font.setPointSizeF(text_font_size);
  painter->setFont(font);

  QFontMetricsF metric(font);
  QString name(inst_->name().c_str());

  QRectF textBBox = metric.boundingRect(name);

  painter->save();

  painter->translate(rect_.left(), rect_.top() + scale_adjust * text_font_size);
  painter->scale(scale_adjust, -scale_adjust);

  QPen pen = painter->pen();
  pen.setColor(color);
  painter->setPen(pen);
  painter->drawText(0, 0, name);

  painter->restore();
}

}
