#ifndef DB_VIAMASTER_H
#define DB_VIAMASTER_H

#include <variant>

#include "dbTechVia.h"
#include "dbGeneratedVia.h"

namespace db
{

template <class... Ts> // Variadic Template
struct overloaded : Ts...
{
  using Ts::operator()...;
};

template <class... Ts> 
overloaded(Ts...) -> overloaded<Ts...>;

class dbViaMaster
{
  public:

    dbViaMaster(dbTechVia* tvia) : via_(tvia) {}
    dbViaMaster(dbGeneratedVia* gvia) : via_(gvia) {}

    // Getters
    const std::variant<dbTechVia*, dbGeneratedVia*>& getVia() const { return via_; }

    // put reference (&) to this return type will cause
    // warning (returning reference to temporary)
    const std::string name() const 
    {
      // Known to be a safe way to traverse std::variant.
      return std::visit( 
          overloaded{[](dbTechVia*      via) { return via->name(); },
                     [](dbGeneratedVia* via) { return via->name(); }}, via_);
    }

  private:

    std::variant<dbTechVia*, dbGeneratedVia*> via_;
};

}

#endif
