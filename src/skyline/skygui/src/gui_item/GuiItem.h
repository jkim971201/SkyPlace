#ifndef GUI_ITEM_H
#define GUI_ITEM_H

#include <memory>

#include <QPainter>
#include <QColor>
#include <QPen>
#include <QBrush>
#include <QGraphicsItem>

#include "LayoutScene.h"

namespace gui
{

class GuiItem : public QGraphicsItem
{
  public:

    GuiItem()
      : isVisible_ (true),
        config_    (nullptr)
    {}

    // Getters
    bool isVisible() const { return isVisible_; }
    std::shared_ptr<GuiConfig> getConfig() { return config_; }
    QPen& getPen() { return pen_; }
    QBrush& getBrush() { return brush_; }

    // Setters
    void setVisible()   { isVisible_ = true;  }
    void setInvisible() { isVisible_ = false; }
    void setConfig(std::shared_ptr<GuiConfig> cfg) { config_ = cfg; }

  protected:

    QPen pen_;
    QBrush brush_;
    bool isVisible_;
    std::shared_ptr<GuiConfig> config_; 
    // don't have to be const, cuz there are not setters in config
};

}

#endif
