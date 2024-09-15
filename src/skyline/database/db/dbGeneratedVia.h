#ifndef DB_GENVIA_H
#define DB_GENVIA_H

#include <string>

namespace db
{

class dbLayer;

// dbGeneratedVia is characterized in the DEF file.
// A generated via is defined using VIARULE.
class dbGeneratedVia
{
  public:

    dbGeneratedVia() {}

    // Setters
    void setName(const char* name) { name_ = std::string(name); }
    void setTopLayer(dbLayer* layer) { topLayer_ = layer; }
    void setCutLayer(dbLayer* layer) { cutLayer_ = layer; }
    void setBotLayer(dbLayer* layer) { botLayer_ = layer; }

    void setCutSizeX(int val) { cutSizeX_ = val; }
    void setCutSizeY(int val) { cutSizeY_ = val; }

    void setCutSpacingX(int val) { cutSpacingX_ = val; }
    void setCutSpacingY(int val) { cutSpacingY_ = val; }

    void setEnclosure(int x1, int y1, int x2, int y2)
    {
      xBotEnc_ = x1;
      yBotEnc_ = y1;
      xTopEnc_ = x2;
      yTopEnc_ = y2;
    }

    void setNumRow(int val) { numRow_ = val; }
    void setNumCol(int val) { numCol_ = val; }

    // Getters
    const std::string& name() const { return name_; }

    int cutSizeX() const { return cutSizeX_; }
    int cutSizeY() const { return cutSizeY_; }

    int cutSpacingX() const { return cutSpacingX_; }
    int cutSpacingY() const { return cutSpacingY_; }

    int xBotEnc() const { return xBotEnc_; }
    int yBotEnc() const { return yBotEnc_; }
    int xTopEnc() const { return xTopEnc_; }
    int yTopEnc() const { return yTopEnc_; }

    int numRow() const { return numRow_; }
    int numCol() const { return numCol_; }

    const dbLayer* getTopLayer() const { return topLayer_; }
          dbLayer* getTopLayer()       { return topLayer_; }

    const dbLayer* getCutLayer() const { return cutLayer_; }
          dbLayer* getCutLayer()       { return cutLayer_; }

    const dbLayer* getBotLayer() const { return botLayer_; }
          dbLayer* getBotLayer()       { return botLayer_; }

  private:

    std::string name_;

    // CUTSIZE
    int cutSizeX_;
    int cutSizeY_;

    // CUTSPACING
    int cutSpacingX_;
    int cutSpacingY_;

    // LAYERS
    dbLayer* topLayer_;
    dbLayer* cutLayer_;
    dbLayer* botLayer_;

    // ENCLOSURE
    int xBotEnc_;
    int yBotEnc_;
    int xTopEnc_;
    int yTopEnc_;

    // ROWCOL
    int numRow_;
    int numCol_;
};

}

#endif
