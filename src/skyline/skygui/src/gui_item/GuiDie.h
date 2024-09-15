#ifndef GUI_DIE_H
#define GUI_DIE_H

#include "GuiItem.h"
#include "db/dbDie.h"

using namespace db;

namespace gui
{

class GuiDie : public GuiItem
{
  public:

    GuiDie(const dbDie* die);

    void setRect(const QRectF& rect) { rect_ = rect; }

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    const dbDie* die_;
    QRectF rect_;
};

}

#endif
