#include <map>
#include <vector>
#include <set>

#include "SkyPlaceDB.h"

namespace skyplace 
{

class Vertex;

class Edge
{
  public:
    Edge(Vertex* v1, Vertex* v2, int weight);

    int weight() const { return weight_; }
    Vertex* v1() const { return v1_;     }
    Vertex* v2() const { return v2_;     }

  private:
    int weight_;

    Vertex* v1_;
    Vertex* v2_;
};

// FS = Total weights of connected-cut-edges
// Weight = Total weights of connected-edges
class Vertex
{
  public:
    Vertex(int id); // id == cell idx

    int       fs() const { return fs_;                             }
    int   weight() const { return weight_;                         }
    int     gain() const { return 2 * fs_ - weight_;               }
    int       id() const { return id_;                             }
    bool isFixed() const { return isFixed_;                        }
    int   degree() const { return static_cast<int>(edges_.size()); }

    void  addEdge(Edge* edge) { edges_.insert(edge); }
    const std::set<Edge*>& edges() const { return edges_; }

    void updateWeight()
    {
      weight_ = 0;
      for(auto e : edges_)
        weight_ += e->weight();
    }

    void setFixed   ()       { isFixed_ = true; }
    void setFS      (int fs) { fs_ = fs;        }
    void setWeight  (int  w) { weight_ = w;     }

  private:
    int fs_; // for FM algorithm
    int weight_;
    int id_; // id == cell idx

    bool isFixed_;
    std::set<Edge*> edges_;
};

class Graph
{
  public:
    Graph(std::vector<Cell*>& cellList);

    const std::vector<Edge*>&      edges() const { return edgePtrs_;    }
    const std::vector<Vertex*>& vertices() const { return vertexPtrs_;  }

    Vertex* getVertex(int id) { return vertexPtrs_[id]; }

  private:
    std::vector<Edge> edges_;
    std::vector<Vertex> vertices_;

    std::vector<Edge*> edgePtrs_;
    std::vector<Vertex*> vertexPtrs_;
};

class FMPartitioner
{
  public:

    FMPartitioner(std::vector<Cell*>& cellList, double balance, int& curNumClutser);

  private:

    void doPartitioning();

    // Bi-Partition
    std::set<Vertex*> part1_;
    std::set<Vertex*> part2_; 

    // Sub-Procedures for FM
    Vertex* findMaxGain(); 
    bool checkBalance(Vertex* v);
    void tryMove(Vertex* v, int& gainSum);
    void realMove(Vertex* v);
    bool checkCut(Edge* e);
    void updateFS(Vertex* v);
    void updateWeight(Vertex* v);
    int  getTotalCut();
    std::vector<std::pair<int, Vertex*>> orderList_;

    std::set<Vertex*> bestPartL_;
    std::set<Vertex*> bestPartU_; 
    std::set<Vertex*> freeVertices_;
    std::set<Vertex*> criticalVertices_;

    double lb_;
    double ub_;
    std::unique_ptr<Graph> graph_;
};

};
