#include "dbTrackGrid.h"

#include <algorithm>

namespace db
{

void
dbTrackGrid::sortTracks()
{
  std::sort(vGrid_.begin(), vGrid_.end(),
  [] (dbTrack& t1, dbTrack& t2) { return t1.start < t2.start; });

  std::sort(hGrid_.begin(), hGrid_.end(),
  [] (dbTrack& t1, dbTrack& t2) { return t1.start < t2.start; });
}

}
