#ifndef PARSER_VERILOG_HPP_
#define PARSER_VERILOG_HPP_

#include <string>
#include <cstddef>
#include <fstream>
#include <variant>
#include <unordered_map>
#include <filesystem>

#include "verilog_scanner.hpp"
#include "verilog_parser.tab.hh"

namespace verilog 
{

class ParserVerilogInterface 
{
  public:
    virtual ~ParserVerilogInterface()
    {
      if(_scanner) delete _scanner;
      if(_parser)  delete _parser;
    }

    virtual void add_module(std::string&&) = 0;
    // port names, begin index, end index, port type (IOB), connection type (wire, reg)
    virtual void add_port(Port&&) = 0;
    virtual void add_net(Net&&) = 0;
    virtual void add_assignment(Assignment&&) = 0;
    virtual void add_instance(Instance&&) = 0;

    bool read(const std::filesystem::path&); 

  private:
    VerilogScanner* _scanner {nullptr};
    VerilogParser*  _parser {nullptr};
};

inline bool ParserVerilogInterface::read(const std::filesystem::path& p)
{
  if(!std::filesystem::exists(p))
    return false;

  std::ifstream ifs(p);

  if(!_scanner)
    _scanner = new VerilogScanner(&ifs);
  if(!_parser)
    _parser = new VerilogParser(*_scanner, this);

  _parser->parse();

  return true;
}

} 

#endif 
