#ifndef PAINTER_H
#define PAINTER_H

#include <vector>
#include <memory>
#include "CImg.h"

#include "SkyPlaceDB.h"

namespace skyplace 
{

static const unsigned char white[]      = {255, 255, 255},
                           black[]      = {  0,   0,   0},
                           red[]        = {255,   0,   0},
                           blue[]       = {120, 200, 255},
                           darkblue[]   = {  0,   0, 255},
                           green[]      = {  0, 255,   0},
                           palegreen[]  = { 30, 240,  30},
                           purple[]     = {255, 100, 255},
                           orange[]     = {255, 165,   0},
                           yellow[]     = {255, 255,   0},
                           darkyellow[] = {240, 190,   0},
                           gray[]       = {204, 204, 204},
                           darkgray[]   = { 40,  40,  40},
                           aqua[]       = {204, 204, 255};

using namespace cimg_library;

typedef const unsigned char* Color;
typedef CImg<unsigned char> CImgObj;

class Painter 
{
  public:

    // Constructor
    Painter() {}

    // APIs
    void drawChip();
    void setDB(std::shared_ptr<SkyPlaceDB> db) { db_ = db; init(); }
    void prepareForPlot();
    void saveImage(int iter, float hpwl, float overflow, bool filler = true);

  private:

    std::shared_ptr<SkyPlaceDB> db_;

    void init();
    void makeDir();

    void show();

    int maxWidth_;
    int maxHeight_;

    int numArrow_;

    int offsetX_;
    int offsetY_;

    int offsetXgif_;
    int offsetYgif_;

    int canvasX_;
    int canvasY_;

    int canvasX_plot_;
    int canvasY_plot_;

    double scale_; 
    double scale_gif_; 
    double scale_window_; 
    // Scaling Factor to fit the window size

    float lenIO_; // Length for IO Triangle

    // Cluster-related
    bool clusterHighlight_;
    std::vector<Color> color_map_; // ClusterID <-> Color
    void genColorMap();

    //CImg library
    CImgObj*       canvas_;
    CImgObj*       canvas_plot_;
    CImgObj*       img_;
    CImgDisplay*   window_;

    std::string outputDir_;

    // Draw Objects
    int getX(int dbX);
    int getY(int dbY);

    int getX(float dbX);
    int getY(float dbY);

    void drawLine(CImgObj *img, int x1, int y1, int x2, int y2);
    void drawLine(CImgObj *img, int x1, int y1, int x2, int y2, Color c);
    void drawRect(CImgObj *img, int lx, int ly, int ux, int uy, Color rect_c, int w);
    void drawRect(CImgObj *img, int lx, int ly, int ux, int uy, Color rect_c, 
                  Color line_c = black, int w = 0, float opacity = 1.0);
                  // line_c: border-line color
                  // w : Thicnkness of border-line

    // Draw Cell (Placer DB)
    void drawCell     (CImgObj *img, const Cell* cell);
    void drawFixed    (CImgObj *img, const Cell* cell);
    void drawFiller   (CImgObj *img, const Cell* cell);
    void drawIO       (CImgObj *img, const Cell* cell, float dieLx, float dieLy, float dieUx, float dieUy); 
    void drawMovable  (CImgObj *img, const Cell* cell);
    void drawCells    (CImgObj *img, const std::vector<Cell*>& cells, bool filler = false);
    void drawDie      (CImgObj *img, const Die*   die);
    void drawCluster  (CImgObj* img, const Cell* cell);

    // Draw Rows
    void drawRows     (CImgObj *img);
    void drawRow      (CImgObj *img, const Row* row);

    // Draw Nets
    void drawNets     (CImgObj *img);
    void drawNet      (CImgObj *img, const Net* net);

    // Draw Bins
    void drawBinsSpectralPotential (CImgObj *img);
    void drawBinsDensity           (CImgObj *img);
    void drawBin                   (CImgObj *img, const Bin* bin, float val, float max, float min);

    // Draw Electric Force (Gradient)
    void drawDensityGradSpectral(CImgObj *img);

    bool check_inside(int lx, int ly, int w, int h);
};

} // namespace skyline

#endif
