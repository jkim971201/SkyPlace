#ifndef DB_WIRE_H
#define DB_WIRE_H

#include <vector>

#include "dbNonDefaultRule.h"

namespace db
{

class dbLayer;
class dbLayerRule;
class dbViaInst;
class dbNet;
class dbWire;

class dbWireSegment 
{
  public:

    dbWireSegment() 
      : viaInst_(nullptr),
        layer_  (nullptr),
        rule_   (nullptr),
        isPatch_(false),
        mask_   (0)
    {}

    // Setters 
    void setStartXY(int x, int y) { startX_ = x; startY_ = y; }
    void setEndXY  (int x, int y) { endX_ = x; endY_ = y; }
    void setRule(dbLayerRule* rule) { rule_ = rule; }
    void setMask(int mask) { mask_ = mask; }
    void setPatch() { isPatch_ = true; }
    void setVia(dbViaInst* viaInst) { viaInst_ = viaInst; }
    void setLayer(dbLayer* layer) { layer_ = layer; }

    // Getters
    bool hasVia()  const { return viaInst_ != nullptr; }
    bool hasRule() const { return rule_ != nullptr; }
    bool isPatch() const { return isPatch_; }
    int  width()   const { return hasRule() ? rule_->width : layer_->width(); }
    int  startX()  const { return startX_;  }
    int  startY()  const { return startY_;  }
    int  endX()    const { return endX_;    }
    int  endY()    const { return endY_;    }
    const dbLayer* layer() const { return layer_; }
    const dbLayerRule* rule() const { return rule_; }
    const dbViaInst* getVia() const { return viaInst_; }

  private:

    int startX_;
    int startY_;
    int endX_;
    int endY_; // vias do not have end point.
    int mask_;
    bool isPatch_; // patch is RECT wire path by Innovus convention.
    dbViaInst* viaInst_;
    dbLayer* layer_; 
    dbLayerRule* rule_;
    // For via, layer_ indicates the bottom layer.
};

class dbWire 
{
  public:

    dbWire() 
      : net_      (nullptr),
        isRouted_ (false)
    {}
  
    void addWireSegment(dbWireSegment* seg) { segments_.push_back(seg); }

    // Setters
    void setRouted() { isRouted_ = true; }
    void setNet(dbNet* net) { net_ = net; }

    // Getters
    bool isRouted() const { return isRouted_; }

    const dbNet* getNet() const { return net_; }
          dbNet* getNet()       { return net_; }

    const std::vector<dbWireSegment*>& getSegments() const { return segments_; }
          std::vector<dbWireSegment*>& getSegments()       { return segments_; }

  private:
    
    dbNet* net_;
    bool isRouted_;
    std::vector<dbWireSegment*> segments_;
};

}

#endif
