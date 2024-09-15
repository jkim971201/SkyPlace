#ifndef GUI_INST_H
#define GUI_INST_H

#include "db/dbInst.h"
#include "GuiItem.h"
#include "GuiPin.h"

using namespace db;

namespace gui
{

class GuiInst : public GuiItem
{
  public:

    GuiInst(const dbInst* inst);

    void setRect(const QRectF& rect) { rect_ = rect; }

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

    void addGuiPin(GuiPin* newPin) { gui_pins_.push_back(newPin); }

  private:

    const dbInst* inst_;
    QRectF rect_;

    std::vector<GuiPin*> gui_pins_;

    void drawInstName(QPainter* painter, const QColor& color, qreal lod);
};

}

#endif
