#ifndef GUI_BLOCKAGE_H
#define GUI_BLOCKAGE_H

#include "db/dbBlockage.h"
#include "GuiRect.h"

using namespace db;

namespace gui
{

class GuiBlockage : public GuiRect
{
  public:

    GuiBlockage(std::shared_ptr<GuiConfig> cfg, 
                const dbBlockage* blk);

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    const dbBlockage* blk_;
};

}

#endif
