#include <iostream>
#include <algorithm>

#include "dbTypes.h"
#include "def/defiComponent.hpp"

namespace db
{

dbTypes::dbTypes()
{
  // Initialization
  str2RoutingType_["CUT"        ] = RoutingType::CUT;
  str2RoutingType_["ROUTING"    ] = RoutingType::ROUTING;
  str2RoutingType_["MASTERSLICE"] = RoutingType::MASTERSLICE;
  str2RoutingType_["OVERLAP"    ] = RoutingType::OVERLAP;
  str2RoutingType_["IMPLANT"    ] = RoutingType::IMPLANT;

  str2LayerDirection_["VERTICAL"  ] = LayerDirection::VERTICAL;
  str2LayerDirection_["HORIZONTAL"] = LayerDirection::HORIZONTAL;

  str2MacroClass_["CORE"            ] = MacroClass::CORE;
  str2MacroClass_["CORE FEEDTHRU"   ] = MacroClass::CORE_FEEDTHRU;
  str2MacroClass_["CORE TIEHIGH"    ] = MacroClass::CORE_TIEHIGH;
  str2MacroClass_["CORE TIELOW"     ] = MacroClass::CORE_TIELOW;
  str2MacroClass_["CORE SPACER"     ] = MacroClass::CORE_SPACER;
  str2MacroClass_["CORE WELLTAP"    ] = MacroClass::CORE_WELLTAP;
  str2MacroClass_["CORE ANTENNACELL"] = MacroClass::CORE_ANTENNACELL;

  str2MacroClass_["PAD"        ] = MacroClass::PAD;
  str2MacroClass_["BLOCK"      ] = MacroClass::BLOCK;
  str2MacroClass_["ENDCAP"     ] = MacroClass::ENDCAP;
  
  str2SiteClass_["CORE"] = SiteClass::CORE_SITE;
  str2SiteClass_["core"] = SiteClass::CORE_SITE;

  str2PinDirection_["INPUT" ] = PinDirection::INPUT;
  str2PinDirection_["OUTPUT"] = PinDirection::OUTPUT;
  str2PinDirection_["INOUT" ] = PinDirection::INOUT;

  str2PinUsage_["SIGNAL" ] = PinUsage::SIGNAL;
  str2PinUsage_["POWER"  ] = PinUsage::POWER;
  str2PinUsage_["GROUND" ] = PinUsage::GROUND;
  str2PinUsage_["CLOCK"  ] = PinUsage::CLOCK;

  str2PinShape_["ABUTMENT"] = PinShape::ABUTMENT;
  str2PinShape_["RING"    ] = PinShape::RING;
  str2PinShape_["FEEDTHRU"] = PinShape::FEEDTHRU;

  str2Orient_["N" ] = Orient::N;
  str2Orient_["S" ] = Orient::S;
  str2Orient_["FN"] = Orient::FN;
  str2Orient_["FS"] = Orient::FS;

  str2Orient_["E" ] = Orient::E;
  str2Orient_["W" ] = Orient::W;
  str2Orient_["FE"] = Orient::FE;
  str2Orient_["FW"] = Orient::FW;

  str2Source_["DIST"   ] = Source::DIST;
  str2Source_["NETLIST"] = Source::NETLIST;
  str2Source_["TIMING" ] = Source::TIMING;
  str2Source_["USER"   ] = Source::USER;

  str2NetUse_["ANALOG"] = NetUse::ANALOG_NET;
  str2NetUse_["CLOCK" ] = NetUse::CLOCK_NET;
  str2NetUse_["GROUND"] = NetUse::GROUND_NET;
  str2NetUse_["POWER" ] = NetUse::POWER_NET;
  str2NetUse_["RESET" ] = NetUse::RESET_NET;
  str2NetUse_["SCAN"  ] = NetUse::SCAN_NET;
  str2NetUse_["SIGNAL"] = NetUse::SIGNAL_NET;
  str2NetUse_["TIEOFF"] = NetUse::TIEOFF_NET;

  str2WireShape_["STRIPE"   ] = WireShape::STRIPE;
  str2WireShape_["FOLLOWPIN"] = WireShape::FOLLOWPIN;

  int2Status_[DEFI_COMPONENT_UNPLACED] = Status::UNPLACED;
  int2Status_[DEFI_COMPONENT_PLACED  ] = Status::PLACED;
  int2Status_[DEFI_COMPONENT_FIXED   ] = Status::FIXED;
  int2Status_[DEFI_COMPONENT_COVER   ] = Status::COVER;
}

RoutingType
dbTypes::getRoutingType(const std::string& str) const
{
  auto itr = str2RoutingType_.find(str);
  
  if(itr == str2RoutingType_.end())
  {
    std::cout << "Error - ROUTING TYPE " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

LayerDirection
dbTypes::getLayerDirection(const std::string& str) const
{
  auto itr = str2LayerDirection_.find(str);
  
  if(itr == str2LayerDirection_.end())
  {
    std::cout << "Error - DIRECTION " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

MacroClass
dbTypes::getMacroClass(const std::string& str) const
{
  auto itr = str2MacroClass_.find(str);
  
  if(itr == str2MacroClass_.end())
  {
    std::cout << "Error - MACRO CLASS " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

SiteClass
dbTypes::getSiteClass(const std::string& str) const
{
  auto itr = str2SiteClass_.find(str);
  
  if(itr == str2SiteClass_.end())
  {
    std::cout << "Error - SITE CLASS " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

PinDirection
dbTypes::getPinDirection(const std::string& str) const
{
  auto itr = str2PinDirection_.find(str);
  
  if(itr == str2PinDirection_.end())
  {
    std::cout << "Error - PIN DIRECTION " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

PinUsage
dbTypes::getPinUsage(const std::string& str) const
{
  auto itr = str2PinUsage_.find(str);
  
  if(itr == str2PinUsage_.end())
  {
    std::cout << "Error - PIN  USAGE " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

PinShape
dbTypes::getPinShape(const std::string& str) const
{
  auto itr = str2PinShape_.find(str);
  
  if(itr == str2PinShape_.end())
  {
    std::cout << "Error - PIN SHAPE " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

Orient
dbTypes::getOrient(const std::string& str) const
{
  auto itr = str2Orient_.find(str);
  
  if(itr == str2Orient_.end())
  {
    std::cout << "Error - Orient " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

Source
dbTypes::getSource(const std::string& str) const
{
  auto itr = str2Source_.find(str);
  
  if(itr == str2Source_.end())
  {
    std::cout << "Error - SOURCE " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

NetUse
dbTypes::getNetUse(const std::string& str) const
{
  auto itr = str2NetUse_.find(str);
  
  if(itr == str2NetUse_.end())
  {
    std::cout << "Error - USE " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

WireShape
dbTypes::getWireShape(const std::string& str) const
{
  auto itr = str2WireShape_.find(str);
  
  if(itr == str2WireShape_.end())
  {
    std::cout << "Error - SHAPE " << str;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

Status
dbTypes::getStatus(int status) const
{
  auto itr = int2Status_.find(status);
  
  if(itr == int2Status_.end())
  {
    std::cout << "Error - PLACEMENT STATUS " << status;
    std::cout << " is unknown (or not supported yet)..." << std::endl;
    exit(0);
  }
  else
    return itr->second;
}

}
