#ifndef LAYOUT_SCENE_H
#define LAYOUT_SCENE_H

#include <array>
#include <memory>
#include <QColor>
#include <QPainter>
#include <QGraphicsScene>

#include "db/dbLayer.h"
#include "db/dbDatabase.h"

using namespace db;

// Follow the color scheme of Innovus 23
// Color name list : w3.org/TR/SVG11/types.html#ColorKeywords
const std::array<QColor, 13> QCOLOR_ARRAY =
{
  QColor(  0,   0, 255), // M1  : blue
  QColor(255,   0,   0), // M2  : red
  QColor(  0, 255, 127), // M3  : spring green
  QColor(139,   0,   0), // M4  : dark red
  QColor(160,  82,  45), // M5  : sienna
  QColor(255, 165,   0), // M6  : orange
  QColor(255,   0, 255), // M7  : magenta
  QColor(  0, 255, 255), // M8  : cyan
  QColor(165,  42,  42), // M9  : brown
  QColor(135, 206, 250), // M10 : lightskyblue
  QColor(255, 255,   0), // M11 : yellow 
  QColor(220,  20,  60), // M12 : crimson
  QColor(255,   0, 255)  // M13 : fuchsia
};

namespace gui
{

class GuiConfig
{
  public:
  
    GuiConfig(std::shared_ptr<dbTech> tech);
    const QColor getLayerColor(const dbLayer* layer);
    int dbu() const { return dbu_; }

    double dieLx = 0.0; // in micron
    double dieLy = 0.0; // in micron
    double dieUx = 0.0; // in micron
    double dieUy = 0.0; // in micron

  private: // should be private?

    int dbu_;
    std::map<const dbLayer*, QColor> layer2Color_;
};

class LayoutScene : public QGraphicsScene
{
  Q_OBJECT

  public:
    
    LayoutScene(QObject* parent = nullptr);

    void setDatabase(std::shared_ptr<dbDatabase> db);
    void createGuiDie();
    void createGuiRow();
    void createGuiInst();
    void createGuiIO();
    void createGuiNet();
    void createGuiBlockage();
    void createGuiTrackGrid();

    void expandScene();

  private:

    std::shared_ptr<dbDatabase> db_;
    std::shared_ptr<GuiConfig> config_;
};

}

#endif
