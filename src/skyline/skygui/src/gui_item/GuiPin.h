#ifndef GUI_PIN_H
#define GUI_PIN_H

#include "GuiItem.h"
#include "db/dbTypes.h"
#include "db/dbInst.h"
#include "db/dbMTerm.h"
#include "db/dbObs.h"

using namespace db;

namespace gui
{

// NOTE : We do not add this item to LayoutScene
class GuiPin : public GuiItem
{
  public:

    GuiPin(std::shared_ptr<GuiConfig> cfg,
           bool isObs,
           const dbInst* inst, 
           const dbLayer* layer,
           const std::vector<std::pair<int, int>>& shape);

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

    bool isObs() const { return isObs_; }

  private:

    // This is for computing boundingRect (BBox)
    double lx_;
    double ly_;
    double ux_;
    double uy_;

    const dbLayer* layer_;
    const dbInst* inst_;

    bool isObs_;
    // To draw POLYGON
    QPolygonF polygon_;
};

}

#endif
