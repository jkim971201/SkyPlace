#include <iostream>
#include <limits>

#include "FMPartitioner.h"

namespace skyplace 
{

// Graph-related
Vertex::Vertex(int id)
{
  id_      = id;
  fs_      = 0;
  weight_  = 0;
  isFixed_ = false;
}

Edge::Edge(Vertex* v1, Vertex* v2, int weight)
{
  v1_     = v1;
  v2_     = v2;
  weight_ = weight;
}

Graph::Graph(std::vector<Cell*>& cellList)
{
  int nVertex = cellList.size();

  int** edge_table = new int*[nVertex];
  for(int i = 0; i < nVertex; i++)
    edge_table[i] = new int[nVertex];

  for(int i = 0; i < nVertex; i++)
  {
    Vertex v(i);
    vertices_.push_back(v);

    for(int j = 0; j < nVertex; j++) 
      edge_table[i][j] = 0; // initialize edge_table
  }

  for(auto& v : vertices_)
    vertexPtrs_.push_back(&v);

  for(int i = 0; i < nVertex - 1; i++)
  {
    Cell* cell1 = cellList[i];
    for(int j = i + 1; j < nVertex; j++)
    {
      if(i != j)
      {
        Cell* cell2 = cellList[j];

        for(auto& pin1 : cell1->pins())
        {
          for(auto& pin2 : pin1->net()->pins())
          {
            if(pin2->isIO())
              continue;
            if(pin2->cell() == cell2)
            {
              //std::cout << "Weight" << std::endl;
              edge_table[i][j] += 1;
              edge_table[j][i] += 1;
            }
          }
        }
      }
    }
  }

  for(int i = 0; i < nVertex - 1; i++)
  {
    for(int j = i + 1; j < nVertex; j++)
    {
      if(edge_table[i][j] != 0)
      {
        Vertex* c = vertexPtrs_[i];
        Vertex* t = vertexPtrs_[j];
        Edge e(c, t, edge_table[i][j]);
        edges_.push_back(e);
      }
    }
  }

  for(auto& e : edges_)
  {
    edgePtrs_.push_back(&e);
    e.v1()->addEdge(&e);
    e.v2()->addEdge(&e);
//    // Debugging
//    Cell* cell1 = cellList[e.v1()->id()];
//    Cell* cell2 = cellList[e.v2()->id()];
  }

  for(auto & v : vertices_)
    v.updateWeight();

  for(int i = 0; i < nVertex; i++)
    delete [] edge_table[i];

  delete [] edge_table;
}

FMPartitioner::FMPartitioner(std::vector<Cell*>& cellList, double balance, int& curNumClutser)
{
  graph_ = std::make_unique<Graph>(cellList);

  lb_ = std::min(balance, 1.0 - balance);
  ub_ = std::max(balance, 1.0 - balance);

  std::cout << "[Partitioner] Number of Edges:    " << graph_->edges().size() << std::endl;
  std::cout << "[Partitioner] Number of Vertices: " << graph_->vertices().size() << std::endl;

  // FM Partitioning
  doPartitioning();

  // Update ClusterID based on partitioning results
  for(auto& v : bestPartL_)
    cellList[v->id()]->setClusterID(curNumClutser);

  //std::cout << "newClusterID : " << curNumClutser << std::endl;

  // Update Clutser Number
  curNumClutser++;

  // Update ClusterID based on partitioning results
  for(auto& v : bestPartU_)
    cellList[v->id()]->setClusterID(curNumClutser);

  //std::cout << "newClusterID : " << curNumClutser << std::endl;

  // Update Clutser Number
  curNumClutser++;

//  if(cellList.size() < 30)
//  {
//    std::cout << "Partition1" << std::endl;
//    
//    for(auto& v : bestPartU_)
//    {
//      Cell* cell = cellList[v->id()];
//      if(cell->bsCellPtr() != nullptr)
//        std::cout << cell->bsCellPtr()->name() << std::endl;
//    }
//
//    std::cout << "Partition2" << std::endl;
//
//    for(auto& v : bestPartL_)
//    {
//      Cell* cell = cellList[v->id()];
//      if(cell->bsCellPtr() != nullptr)
//        std::cout << cell->bsCellPtr()->name() << std::endl;
//    }
//  }
}

void
FMPartitioner::doPartitioning()
{
  part1_.clear();
  part2_.clear();

  // Step1: Initial Partitioning -> from Spectral Partitioning
  int numVertex = graph_->vertices().size();
  for(auto& v : graph_->vertices())
  {
    if(v->id() < numVertex / 2 ) 
      part1_.insert(v);
    else                                       
      part2_.insert(v);
  }

  bestPartL_ = part1_;
  bestPartU_ = part2_;

  // Step2: Initialization
  for(auto& v : graph_->vertices())
  {
    freeVertices_.insert(v);
    updateFS(v);
    updateWeight(v);
  }

  int gainSum = 0;

  std::cout << "[Partitioner] Start Cut : " << getTotalCut() << std::endl;

  // Step3: Main Loop
  while(!freeVertices_.empty())
  {
    Vertex* maxVertex = nullptr;
    maxVertex = findMaxGain();

    if(maxVertex) tryMove(maxVertex, gainSum);
    else          break;

    orderList_.push_back(std::make_pair(gainSum, maxVertex));

    //if(maxVertex)
    //  printf("MaxVertex : %d gainSum : %d\n", maxVertex->id(), gainSum);

    for(auto& v : criticalVertices_)
    {
      updateFS(v);
      updateWeight(v);
    }
  }

  // Step4: Find the best move in the OrderList
  int bestGainSum = 0;
  int bestGainIdx = 0;
  for(size_t i = 0; i < orderList_.size(); i++)
  {
    if(orderList_[i].first > bestGainSum)
    {
      bestGainIdx = i + 1;
      bestGainSum = orderList_[i].first;
    }
  }

  // Step5: Execute the best move
  for(size_t i = 0; i < bestGainIdx; i++)
  {
    Vertex* v = orderList_[i].second;  
    realMove(v);
    // printf("Debug) Real Move %d \n", v->id());
  }

  // Step6: Make the Final partitions
  part1_ = bestPartL_;
  part2_ = bestPartU_;

  std::cout << "[Partitioner] Final Cut : " << getTotalCut() << std::endl;
}

Vertex*
FMPartitioner::findMaxGain()
{
  int MAX_GAIN = std::numeric_limits<int>::min();
  Vertex* maxVertex = nullptr;

  for(auto& v : graph_->vertices())
  {
    bool breakTie = false;

    if(checkBalance(v) && !v->isFixed())
    {
      if(v->gain() >= MAX_GAIN) 
        breakTie = true;
      else if(v->gain() == MAX_GAIN)
      {
        // Tie-Breaking Policy
        int sizeL = static_cast<int>(part1_.size());
        int sizeU = static_cast<int>(part2_.size());

        if(part1_.count(v))
        {
          if(sizeL > sizeU) breakTie = true;
        }
        else if(part2_.count(v))
        {
          if(sizeU > sizeL) breakTie = true;
        }
      }

      if(breakTie)
      {
        MAX_GAIN = v->gain();
        maxVertex = v;  
      }
    }
  }

  return maxVertex;
}

bool
FMPartitioner::checkBalance(Vertex* v)
{
  int sizeL = static_cast<int>(part1_.size());
  int sizeU = static_cast<int>(part2_.size());

  int total = sizeL + sizeU;

  if(part1_.count(v))
  {
    double lb = static_cast<double>(std::min(sizeL - 1, sizeU + 1)) 
              / static_cast<double>(total);   

    double ub = static_cast<double>(std::max(sizeL - 1, sizeU + 1)) 
              / static_cast<double>(total);   

    if(lb < lb_ || ub > ub_) return false;
  }
  else if(part2_.count(v))
  {
    double lb = static_cast<double>(std::min(sizeL + 1, sizeU - 1)) 
              / static_cast<double>(total);   

    double ub = static_cast<double>(std::max(sizeL + 1, sizeU - 1)) 
              / static_cast<double>(total);   

    if(lb < lb_ || ub > ub_) return false;
  }

  return true;
}

void
FMPartitioner::tryMove(Vertex* v, int& gainSum)
{
  criticalVertices_.clear();

  if(part1_.count(v))
  {
    part1_.erase(v);
    part2_.insert(v);
  }
  else if(part2_.count(v))
  {
    part2_.erase(v);
    part1_.insert(v);
  }

  for(auto& e : v->edges())
  {
    criticalVertices_.insert(e->v1());
    criticalVertices_.insert(e->v2());
  }

  v->setFixed();
  freeVertices_.erase(v);
  gainSum += v->gain();
}

void
FMPartitioner::realMove(Vertex* v)
{
  criticalVertices_.clear();

  if(bestPartL_.count(v))
  {
    bestPartL_.erase(v);
    bestPartU_.insert(v);
  }
  else if(bestPartU_.count(v))
  {
    bestPartU_.erase(v);
    bestPartL_.insert(v);
  }
}

bool
FMPartitioner::checkCut(Edge* e)
{
  bool isCut = false;

  if(part1_.count( e->v1() ) )
  {
    if(part1_.count( e->v2() ) ) isCut = false;
    else                         isCut = true;
  }
  else if(part2_.count( e->v1() ) )
  {
    if(part2_.count( e->v2() ) ) isCut = false;
    else                         isCut = true;
  }

  //printf("Vertex1: %d Vertex2: %d isCut? : %d\n", 
  //        e->v1()->id(), e->v2()->id(), isCut);

  return isCut;
}

// FS = Total weights of connected-cut-edges
void 
FMPartitioner::updateFS(Vertex* v)
{
  int fs = 0;
  for(auto& e : v->edges())
  {
    if(checkCut(e)) 
      fs += e->weight();
  }

  v->setFS(fs);
}

// Weight = Total weights of connected-edges
void 
FMPartitioner::updateWeight(Vertex* v)
{
  int w = 0;
  for(auto& e : v->edges())
    w += e->weight();

  v->setWeight(w);
}

int
FMPartitioner::getTotalCut()
{
  int cut = 0;
  for(auto e : graph_->edges())
  {
    if(checkCut(e)) 
      cut += e->weight();
  }

  return cut;
}

};
