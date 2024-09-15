#ifndef GUI_IO_H
#define GUI_IO_H

#include "db/dbTypes.h"
#include "db/dbBTerm.h"
#include "GuiItem.h"
#include "GuiRect.h"

using namespace db;

namespace gui
{

class GuiIOPort : public GuiRect
{
  public:

    GuiIOPort(std::shared_ptr<GuiConfig> cfg, 
              const dbBTermPort* port);

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    double lx_;
    double ly_;
    double ux_;
    double uy_;

    const dbBTermPort* port_;
};

class GuiIO : public GuiItem
{
  public:

    GuiIO(std::shared_ptr<GuiConfig> cfg, 
          const dbBTerm* io);

    QRectF boundingRect() const override;

    void paint(QPainter* painter, 
               const QStyleOptionGraphicsItem* option, 
               QWidget* widget) override;

  private:

    const dbBTerm* io_;

    double lx_;
    double ly_;
    double ux_;
    double uy_;

    std::vector<GuiRect*> gui_ports_;
};

}

#endif
