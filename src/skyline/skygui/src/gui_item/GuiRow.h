#ifndef GUI_ROW_H
#define GUI_ROW_H

#include "GuiItem.h"
#include "db/dbRow.h"

using namespace db;

namespace gui
{

class GuiRow : public GuiItem
{
  public:

    GuiRow(dbRow* row);

    void setRect(const QRectF& rect) { rect_ = rect; }

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    dbRow* row_;
    QRectF rect_;
};

}

#endif
