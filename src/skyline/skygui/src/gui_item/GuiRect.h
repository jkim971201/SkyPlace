#ifndef GUI_RECT_H
#define GUI_RECT_H

#include "db/dbRect.h"
#include "GuiItem.h"

using namespace db;

namespace gui
{

class GuiRect : public GuiItem 
{
  public:

    GuiRect();
    GuiRect(double lx, double ly, double dx, double dy);

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  protected:

    QRectF rect_;
};

}

#endif
