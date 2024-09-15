#ifndef DB_TECHVIA_H
#define DB_TECHVIA_H

#include <string>
#include <vector>

#include "dbLayer.h"
#include "dbRect.h"
#include "dbPolygon.h"

namespace db
{

class dbRect;
class dbPolygon;

// dbTechVia is characterized in the LEF file.
class dbTechVia
{
  public:

    dbTechVia();

    void print() const;
    void setBoundary();

    // Setters
    void setName(const char* name) { name_ = std::string(name); }
    void setResistance(double res) { res_ = res; }
    void setDefault() { isDefault_ = true; }
    void setTopLayer(dbLayer* layer) { topLayer_ = layer; }
    void setCutLayer(dbLayer* layer) { cutLayer_ = layer; }
    void setBotLayer(dbLayer* layer) { botLayer_ = layer; }
    void addRect(dbRect* rect) { rects_.push_back(rect); }
    void addPolygon(dbPolygon* poly) { polygons_.push_back(poly); }

    // Getters
    const std::vector<std::pair<int, int>>& getCutLayerShape() const { return shape_; }

    const std::string& name() const { return name_; }

    const dbLayer* getTopLayer() const { return topLayer_; }
          dbLayer* getTopLayer()       { return topLayer_; }

    const dbLayer* getCutLayer() const { return cutLayer_; }
          dbLayer* getCutLayer()       { return cutLayer_; }

    const dbLayer* getBotLayer() const { return botLayer_; }
          dbLayer* getBotLayer()       { return botLayer_; }

    bool isDefault() const { return isDefault_; }
    bool hasViaRule() const { return hasViaRule_; }

    const std::vector<dbRect*>& getRects() const { return rects_; }
    const std::vector<dbPolygon*>& getPolygons() const { return polygons_; }

  private:

    std::string name_;

    double res_;
    
    bool isDefault_;
    bool hasViaRule_;
    // Not supported yet

    dbLayer* topLayer_;
    dbLayer* cutLayer_;
    dbLayer* botLayer_;

    std::vector<dbRect*>    rects_;
    std::vector<dbPolygon*> polygons_;

    // Shape of Cut Layer
    std::vector<std::pair<int, int>> shape_;
};

}

#endif
