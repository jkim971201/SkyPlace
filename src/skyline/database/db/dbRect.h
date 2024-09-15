#ifndef DB_RECT_H
#define DB_RECT_H

namespace db
{

class dbLayer;

// This class is used when
// #1. for describing the pin shape of dbMTerm (LEF PIN)
// #2. for describing the obstruct shape       (LEF OBS)
// #3. for describing the boundary of dbBTerm  (DEF PIN PORT)
class dbRect
{
  public:

  dbRect()
    : lx_    (0),
      ly_    (0),
      ux_    (0),
      uy_    (0),
      layer_ (nullptr)
  {}

  dbRect(int lx, int ly, int ux, int uy, dbLayer* _layer)
    : lx_    (lx), 
      ly_    (ly),
      ux_    (ux), 
      uy_    (uy), 
      layer_ (_layer) 
  {}

  // Setters
  void setLx(int val) { lx_ = val; }
  void setLy(int val) { ly_ = val; }
  void setUx(int val) { ux_ = val; }
  void setUy(int val) { uy_ = val; }
  void setLayer(dbLayer* layer) { layer_ = layer; }

  // Getters
  int ux() const { return ux_; }
  int uy() const { return uy_; }
  int lx() const { return lx_; }
  int ly() const { return ly_; }
  int dx() const { return ux_ - lx_; }
  int dy() const { return uy_ - ly_; }
  int cx() const { return (ux_ + lx_) / 2; }
  int cy() const { return (uy_ + ly_) / 2; }

  int64_t area() const { return static_cast<int64_t>( dx() ) 
                              * static_cast<int64_t>( dy() ); }

  const dbLayer* layer() const { return layer_; }
        dbLayer* layer()       { return layer_; }

  protected:

    int lx_;
    int ly_;
    int ux_;
    int uy_;
    dbLayer* layer_;
};

}

#endif
