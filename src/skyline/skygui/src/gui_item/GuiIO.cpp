#include <cassert>
#include <iostream>

#include <QStyleOptionGraphicsItem>
#include "GuiIO.h"

namespace gui
{

GuiIOPort::GuiIOPort(std::shared_ptr<GuiConfig> cfg,
                     const dbBTermPort* port)
  : port_(port)
{
  setConfig(cfg);

  const double dbu = static_cast<double>(getConfig()->dbu());

  int ioLx_dbu = port_->lx();
  int ioLy_dbu = port_->ly();
  int ioUx_dbu = port_->ux();
  int ioUy_dbu = port_->uy();
  int ioDx_dbu = port_->dx();
  int ioDy_dbu = port_->dy();

  double ioLx_micron = ioLx_dbu / dbu;
  double ioLy_micron = ioLy_dbu / dbu;
  double ioUx_micron = ioUx_dbu / dbu;
  double ioUy_micron = ioUy_dbu / dbu;
  double ioDx_micron = ioDx_dbu / dbu;
  double ioDy_micron = ioDy_dbu / dbu;

  lx_ = ioLx_micron;
  ly_ = ioLy_micron;
  ux_ = ioUx_micron;
  uy_ = ioUy_micron;

  rect_ = QRectF(lx_, ly_, ux_ - lx_, uy_ - ly_);
}

void
GuiIOPort::paint(QPainter* painter, 
                 const QStyleOptionGraphicsItem* option,
                 QWidget* widget)
{
  const qreal lod 
    = option->levelOfDetailFromTransform(painter->worldTransform());
  
  const auto layer = port_->layer();
  const auto color = getConfig()->getLayerColor(layer);

  getPen().setColor(color);
  getBrush().setColor(color);
  //getBrush().setStyle(Qt::BrushStyle::DiagCrossPattern);
  getBrush().setStyle(Qt::BrushStyle::Dense1Pattern);

  GuiRect::paint(painter, option, widget);

  qreal ioLx = rect_.left();
  qreal ioLy = rect_.top();
  qreal ioUx = rect_.right();
  qreal ioUy = rect_.bottom();

  qreal len = std::min(std::abs(rect_.width()), 
                           std::abs(rect_.height())) * 2.0;

  if(lod < 1.0)
    len = len / (lod * 0.1);
  else if(lod > 1.0 && lod < 10.0)
    len = len * 10.0;
  else 
    len = len;

  qreal p1X = 0.0;
  qreal p1Y = 0.0;
  qreal p2X = 0.0;
  qreal p2Y = 0.0;
  qreal p3X = 0.0;
  qreal p3Y = 0.0;

  // Couter clock-wise
  // 0.8660 ~= sqrt(3)/2
  switch(port_->orient())
  {
    case Orient::N :
    {
      p1X = ioLx;
      p1Y = ioLy;
      p2X = p1X - len * 0.5;
      p2Y = p1Y - len * 0.8660;
      p3X = p1X + len * 0.5;
      p3Y = p1Y - len * 0.8660;

      lx_ -= len * 0.5;
      ly_ -= len * 0.8660;
      break;
    }
    case Orient::S :
    {
      p1X = ioUx;
      p1Y = ioUy;
      p2X = p1X + len * 0.5;
      p2Y = p1Y + len * 0.8660;
      p3X = p1X - len * 0.5;
      p3Y = p1Y + len * 0.8660;

      ux_ += len * 0.5;
      uy_ += len * 0.8660;
      break;
    }
    case Orient::W :
    {
      p1X = ioUx;
      p1Y = ioLy;
      p2X = p1X + len * 0.8660;
      p2Y = p1Y - len * 0.5;
      p3X = p1X + len * 0.8660;
      p3Y = p1Y + len * 0.5;

      ux_ += len * 0.8660;
      ly_ -= len * 0.5;
      break;
    }
    case Orient::E :
    {
      p1X = ioLx;
      p1Y = ioUy;
      p2X = p1X - len * 0.8660;
      p2Y = p1Y + len * 0.5;
      p3X = p1X - len * 0.8660;
      p3Y = p1Y - len * 0.5;

      lx_ -= len * 0.8660;
      uy_ += len * 0.5;
      break;
    }
    case Orient::FN :
    {
      p1X = ioUx;
      p1Y = ioLy;
      p2X = p1X - len * 0.5;
      p2Y = p1Y - len * 0.8660;
      p3X = p1X + len * 0.5;
      p3Y = p1Y - len * 0.8660;

      ux_ += len * 0.5;
      ly_ -= len * 0.8660;
      break;
    }
    case Orient::FS :
    {
      p1X = ioLx;
      p1Y = ioUy;
      p2X = p1X + len * 0.5;
      p2Y = p1Y + len * 0.8660;
      p3X = p1X - len * 0.5;
      p3Y = p1Y + len * 0.8660;

      ux_ += len * 0.5;
      ly_ -= len * 0.8660;
      break;
    }
    case Orient::FW :
    {
      p1X = ioLx;
      p1Y = ioLy;
      p2X = p1X - len * 0.8660;
      p2Y = p1Y + len * 0.5;
      p3X = p1X - len * 0.8660;
      p3Y = p1Y - len * 0.5;

      lx_ -= len * 0.8660;
      ly_ -= len * 0.5;
      break;
    }
    case Orient::FE :
    {
      p1X = ioUx;
      p1Y = ioUy;
      p2X = p1X + len * 0.8660;
      p2Y = p1Y - len * 0.5;
      p3X = p1X + len * 0.8660;
      p3Y = p1Y + len * 0.5;

      ux_ += len * 0.8660;
      uy_ += len * 0.5;
      break;
    }
    default:
      break;
  }

  getPen().setColor(QColor(255, 215, 0)); // color name : gold
  getBrush().setColor(QColor(255, 215, 0)); 
  getBrush().setStyle(Qt::BrushStyle::SolidPattern);

  painter->setPen(getPen());
  painter->setBrush(getBrush());

  QPainterPath path;
  path.moveTo(p1X, p1Y);
  path.lineTo(p2X, p2Y);
  path.lineTo(p3X, p3Y);
  path.lineTo(p1X, p1Y);

  painter->drawPath(path);
}

QRectF
GuiIOPort::boundingRect() const
{
  return rect_;
}

/* GuiIO */
GuiIO::GuiIO(std::shared_ptr<GuiConfig> cfg,
             const dbBTerm* io)
  : io_(io)
{
  setConfig(cfg);

  lx_ = std::numeric_limits<double>::max();
  ly_ = std::numeric_limits<double>::max();
  ux_ = std::numeric_limits<double>::min();
  uy_ = std::numeric_limits<double>::min();

  const double dbu = static_cast<double>(config_->dbu());

  for(const auto port : io_->ports())
  {
    GuiIOPort* gui_port 
      = new GuiIOPort(config_, port);

    gui_ports_.push_back(gui_port);

    const auto bbox = gui_port->boundingRect();

    lx_ = std::min(lx_, bbox.left());
    ly_ = std::min(ly_, bbox.bottom());
    ux_ = std::max(ux_, bbox.right());
    uy_ = std::max(uy_, bbox.top());
  }
}

QRectF
GuiIO::boundingRect() const
{
  // QGrahicsView decide to re-paint a QGraphicsItem when
  // its boundingRect() is inside the scene.
  // If boundingRect() is not computed properly,
  // an item can disappear while repainting.
  return QRectF(lx_, ly_, ux_ - lx_, uy_ - ly_);
}

void
GuiIO::paint(QPainter* painter, 
             const QStyleOptionGraphicsItem* option,
             QWidget* widget)
{
  for(auto port : gui_ports_)
    port->paint(painter, option, widget);
}

}
