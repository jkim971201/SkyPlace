#ifndef DB_TRACKGRID_H
#define DB_TRACKGRID_H

#include <vector>

namespace db
{

class dbLayer;

struct dbTrack
{
  int start = 0;
  int spacing = 0;;
};

class dbTrackGrid
{
  public:

    dbTrackGrid() : layer_(nullptr) {}

    // Sort tracks in increasing order of starting coordinates.
    // + check if satisfying spacing rule.
    void sortTracks();

    void addVTrack(dbTrack t) { vGrid_.push_back(t); }
    void addHTrack(dbTrack t) { hGrid_.push_back(t); }

    // Setters
    void setLayer(dbLayer* layer) { layer_ = layer; }

    // Getters
    const dbLayer* layer() const { return layer_; }

    const std::vector<dbTrack>& getVGrid() const { return vGrid_; }
    const std::vector<dbTrack>& getHGrid() const { return hGrid_; }

  private:

    dbLayer* layer_;
    std::vector<dbTrack> vGrid_;
    std::vector<dbTrack> hGrid_;
};

}

#endif
