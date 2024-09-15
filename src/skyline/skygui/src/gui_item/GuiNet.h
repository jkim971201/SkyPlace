#ifndef GUI_NET_H
#define GUI_NET_H

#include "db/dbNet.h"
#include "db/dbWire.h"
#include "GuiItem.h"
#include "GuiRect.h"

using namespace db;

namespace gui
{

// NOTE : We do not add this item to LayoutScene
class GuiWireSegment : public GuiRect
{
  public:

    GuiWireSegment(std::shared_ptr<GuiConfig> cfg,
                   const dbWireSegment* seg);

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    const dbWireSegment* seg_;
};

class GuiNet : public GuiItem
{
  public:

    GuiNet(std::shared_ptr<GuiConfig> cfg,
           const dbNet* net);

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    const dbNet* net_;
    QRectF rect_;

    std::vector<GuiWireSegment*> gui_wires_;
};

}

#endif
