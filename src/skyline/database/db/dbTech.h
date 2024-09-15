#ifndef DB_TECH
#define DB_TECH

#include <memory>
#include <string>
#include <unordered_map>

#include "lef/lefrReader.hpp"
#include "dbTypes.h"
#include "dbLayer.h"
#include "dbSite.h"
#include "dbMacro.h"
#include "dbViaMaster.h"

namespace db
{

class dbTech
{
  public:

    dbTech(std::shared_ptr<dbTypes> types);
    ~dbTech();

    void setUnits        (const lefiUnits* unit);
    void setBusBit       (const char* busBit);
    void setDivider      (const char divider);
    void createNewLayer  (const lefiLayer* la);
    void createNewSite   (const lefiSite* si);
    void createNewVia    (const lefiVia* via);
    void addPinToMacro   (const lefiPin*         pi, dbMacro* topMacro);
    void addObsToMacro   (const lefiObstruction* ob, dbMacro* topMacro);

    dbMacro* getNewMacro (const char* name);
    void fillNewMacro    (const lefiMacro* ma, dbMacro* newMacro); 

    void setDbu(int dbu) { dbu_ = dbu; }
    int  getDbu() const { return dbu_; }
    int  getDbuLength(double micron) const;
    int  getDbuArea  (double micron) const;

    const char getDivider() const { return divider_;             }
    int getRightBusBit()    const { return right_bus_delimiter_; }
    int getLeftBusBit()     const { return left_bus_delimiter_;  }

    dbSite*      getSiteByName     (const std::string& name);
    dbMacro*     getMacroByName    (const std::string& name);
    dbLayer*     getLayerByName    (const std::string& name);
    dbViaMaster* getViaMasterByName(const std::string& name);

    const std::vector<dbLayer*>&     getLayers()     const { return layers_; }
    const std::vector<dbViaMaster*>& getViaMasters() const { return vias_; }
    const std::vector<dbSite*>&      getSites()      const { return sites_; }

  private:

    // DBU per MICRON
    int dbu_;

    char left_bus_delimiter_;
    char right_bus_delimiter_;
    char divider_;

    std::shared_ptr<dbTypes> types_;

    std::unordered_map<std::string, dbLayer*>     str2dbLayer_;
    std::unordered_map<std::string, dbViaMaster*> str2dbViaMaster_;
    std::unordered_map<std::string, dbSite*>      str2dbSite_;
    std::unordered_map<std::string, dbMacro*>     str2dbMacro_;

    std::vector<dbLayer*>     layers_;
    std::vector<dbSite*>      sites_;
    std::vector<dbMacro*>     macros_;
    std::vector<dbViaMaster*> vias_;
};

}

#endif
