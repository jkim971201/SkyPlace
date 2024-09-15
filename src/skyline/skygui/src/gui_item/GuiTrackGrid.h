#ifndef GUI_TRACKGRID_H
#define GUI_TRACKGRID_H

#include "GuiItem.h"
#include "db/dbTrackGrid.h"

#include <vector>

using namespace db;

namespace gui
{

class GuiTrackGrid: public GuiItem
{
  public:

    GuiTrackGrid(std::shared_ptr<GuiConfig> cfg,
                 const dbTrackGrid* grid);

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    const dbTrackGrid* grid_;
    QRectF rect_;

    std::vector<QLineF> gui_tracks_;
};

}

#endif
