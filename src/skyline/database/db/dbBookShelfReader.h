#ifndef DB_BOOKSHELF_READER_H
#define DB_BOOKSHELF_READER_H

#include <memory>

#include "BookShelfParser.h"
#include "dbDesign.h"

using namespace bookshelf;

namespace db
{

class dbBookShelfReader
{
  public:

    dbBookShelfReader(std::shared_ptr<dbTypes>  types,
                      std::shared_ptr<dbDesign> design);

    void readFile(const std::string& filename);

    int dbuBookShelf() const { return dbuBookShelf_; }

  private:

    void convert2db();

    int dbuBookShelf_;

    std::shared_ptr<dbTypes>         types_;
    std::shared_ptr<dbDesign>        design_;
    std::unique_ptr<BookShelfParser> bsParser_;
};

}

#endif
