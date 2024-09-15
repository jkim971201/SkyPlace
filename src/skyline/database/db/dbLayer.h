#ifndef DB_LAYER_H
#define DB_LAYER_H

#include <string>

#include "dbTypes.h"

namespace db
{

// TODO : Add Width Table
class dbLayer
{

  public:

    dbLayer();

    void print() const;

    // Setters
    void setName      (const char*   name) { name_ = std::string(name); }
    void setIndex     (int            val) { index_   = val;  }
    void setXPitch    (int            val) { xPitch_  = val;  }
    void setYPitch    (int            val) { yPitch_  = val;  }
    void setXOffset   (int            val) { xOffset_ = val;  }
    void setYOffset   (int            val) { yOffset_ = val;  }
    void setWidth     (int            val) { width_   = val;  }
    void setSpacing   (int            val) { spacing_ = val;  }
    void setArea      (double         val) { area_    = val;  }
    void setType      (RoutingType   type) { type_    = type; }
    void setDirection (LayerDirection dir) { dir_     = dir;  }

    // Getters
    int xPitch()  const { return xPitch_;  }
    int yPitch()  const { return yPitch_;  }
    int xOffset() const { return xOffset_; }
    int yOffset() const { return yOffset_; }
    int width()   const { return width_;   }
    int spacing() const { return spacing_; }
    int area()    const { return area_;    }
    int index()   const { return index_;   }
    // Index will be used to identify vertical order of layers.
    // e.g. layer that has 0 index is the bottom layer.

    RoutingType    type()      const { return type_; }
    LayerDirection direction() const { return dir_;  }

    const std::string& name() const { return name_; }

  private:

    // LEF Syntax
    std::string name_;

    int xPitch_;
    int yPitch_;
    int xOffset_;
    int yOffset_;
    int width_;
    int spacing_;
    int area_;
    int index_;

    RoutingType type_;
    LayerDirection dir_;
};

}

#endif
