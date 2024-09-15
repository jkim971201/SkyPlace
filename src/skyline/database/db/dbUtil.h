#ifndef DB_UTIL_H
#define DB_UTIL_H

#include <cstdio>
#include <ctime>
#include <chrono> // To obtain date info.
#include <string>
#include <unordered_map>

namespace db
{

template <typename M>
inline bool duplicateCheck(M& map, const std::string& name)
{
  // true -> already exists
  if(map.find(name) != map.end())
    return true;
  else return false;
}

inline std::string removeBackSlashBracket(const std::string& str)
{
  std::string newStr = str;
  //printf("before : %s\n", newStr.c_str());
  if(newStr.find("\\[") != std::string::npos && newStr.find("\\]") != std::string::npos)
  {
    size_t bracket1 = newStr.find("\\[");
    while(bracket1 != std::string::npos)
    {
      newStr.erase(newStr.begin() + bracket1);
      bracket1 = newStr.find("\\[");
    }

    size_t bracket2 = newStr.find("\\]");
    while(bracket2 != std::string::npos)
    {
      newStr.erase(newStr.begin() + bracket2);
      bracket2 = newStr.find("\\]");
    }
  }
  //printf("after : %s\n", newStr.c_str());

  return newStr;
}

inline std::string getCalenderDate()
{
  auto now = std::chrono::system_clock::now();
  std::time_t now_ctime = std::chrono::system_clock::to_time_t(now);

  auto parts = std::localtime(&now_ctime);

  int year = parts->tm_year + 1900;
  int mon  = parts->tm_mon + 1;
  int day  = parts->tm_mday;
  
  std::string calenderDate =
      std::to_string(year) + "-" 
    + std::to_string(mon) + "-" 
    + std::to_string(day);

  return calenderDate;
}

inline std::string getClockTime()
{
  auto now = std::chrono::system_clock::now();
  std::time_t now_ctime = std::chrono::system_clock::to_time_t(now);

  auto parts = std::localtime(&now_ctime);

  int hour = parts->tm_hour;
  int min  = parts->tm_min;
  int sec  = parts->tm_sec;
  
  std::string clockTime =
      std::to_string(hour) + ":" 
    + std::to_string(min) + ":" 
    + std::to_string(sec);

  return clockTime;
}

inline void techNotExist(const char* type, const char* name)
{
  printf("Error - %s %s", type, name);
  printf(" does not exist in the technology database...\n");
  exit(1);
}

inline void designNotExist(const char* type, const char* name)
{
  printf("Error - %s %s", type, name);
  printf(" does not exist in the design database...\n");
  exit(1);
}

inline void alreadyExist(const char* type, const char* name)
{
  printf("Error - %s %s already exists...\n", type, name);
  exit(1);
}

inline void unsupportSyntax(const char* syntax)
{
  printf("%s is not supported syntax...\n", syntax);
  exit(1);
}

}

#endif
