#ifndef DB_DATABASE_H
#define DB_DATABASE_H

#include <set>
#include <string>
#include <filesystem>
#include <memory>

namespace db
{

class dbTypes;
class dbTech;
class dbDesign;
class dbLefReader;
class dbDefReader;
class dbVerilogReader;
class dbBookShelfReader;

class dbDatabase
{
  public:

    dbDatabase();

    ~dbDatabase() {}

    void readLef          (const char* filename);
    void readDef          (const char* filename);
    void readVerilog      (const char* filename);
    void readBookShelf    (const char* filename);

    void writeBookShelf   (const char* filename) const;
    void writeDef         (const char* filename) const;

    void setTopModuleName (const char*  topname);

    std::shared_ptr<dbTech>   getTech()   { return tech_;   }
    std::shared_ptr<dbDesign> getDesign() { return design_; }

    bool isBookShelf() const { return bookShelfFlag_; }

  private:

    bool bookShelfFlag_;

    // Parsing
    std::shared_ptr<dbLefReader>       lefReader_;
    std::shared_ptr<dbDefReader>       defReader_;
    std::shared_ptr<dbVerilogReader>   verilogReader_;
    std::shared_ptr<dbBookShelfReader> bsReader_;
    
    std::string auxFile_;           // File name  of .aux already read
    std::string   vFile_;           // File name  of .v   already read
    std::string defFile_;           // File name  of .def already read
    std::set<std::string> lefList_; // File names of .lef already read

    // Technology (PDK) and Design
    std::shared_ptr<dbTech>   tech_;
    std::shared_ptr<dbTypes>  types_;
    std::shared_ptr<dbDesign> design_;
};

}

#endif
