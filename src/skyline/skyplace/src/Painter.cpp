#include <string>
#include <iostream>
#include <climits>   // For INT_MAX, INT_MIN
#include <cfloat>    // For FLT_MAX
#include <cmath>
#include <random>
#include <sstream> // for ostringstream
#include <iomanip> // for setw, setfill

#include "Painter.h"

// Not a real size
// MAX_W, MAX_H is just a imaginary size
#define MAX_W 6000
#define MAX_H 6000

#define MAX_W_PLOT 800 // 2000
#define MAX_H_PLOT 800 // 2000

#define WINDOW_W 1600
#define WINDOW_H 1600

#define ZOOM_SPEED 300
#define MOVE_SPEED 300

#define DIE_OFFSET_X 10
#define DIE_OFFSET_Y 10

#define DIE_OPACITY         1.0
#define MACRO_OPACITY       0.8
#define STD_CELL_OPACITY    0.8
#define FIXED_CELL_OPACITY  0.8
#define FILLER_CELL_OPACITY 0.8
#define IO_OPACITY          0.4

#define DIE_LINE_THICKNESS      1
#define MACRO_LINE_THICKNESS    1
#define STD_CELL_LINE_THICKNESS 0
#define IO_LINE_THICKNESS       0

namespace skyplace 
{

using namespace cimg_library;

static const Color BACKGROND_COLOR       = black;
static const Color DIE_COLOR             = darkgray; // darkgray
static const Color MACRO_COLOR           = red; // red
static const Color FIXED_CELL_COLOR      = red; // aqua
static const Color STD_CELL_COLOR        = gray; // gray
static const Color FILLER_CELL_COLOR     = blue;
static const Color IO_COLOR              = yellow;

static const Color NET_LINE_COLOR        = blue;
static const Color DIE_LINE_COLOR        = darkgray;
static const Color MACRO_LINE_COLOR      = black;
static const Color STD_CELL_LINE_COLOR   = STD_CELL_COLOR; 
// for gif plot mode, same color will look better 
static const Color FILLER_LINE_COLOR     = blue;
static const Color IO_LINE_COLOR         = IO_COLOR;
static const Color ROW_LINE_COLOR        = palegreen;

static const Color ARROW_COLOR           = orange;

inline void interpolate(float value, 
                        int r1, 
                        int g1, 
                        int b1,
                        int r2, 
                        int g2, 
                        int b2,
                        int& r, 
                        int& g, 
                        int& b)
{
  r = static_cast<int>(std::abs(r2 - r1) * value + static_cast<float>(r1) );
  g = static_cast<int>(std::abs(g2 - g1) * value + static_cast<float>(g1) );
  b = static_cast<int>(std::abs(b2 - b1) * value + static_cast<float>(b1) );
}

void setRGB(int& r, int &g, int &b, float val, float max, float min)
{
  float ratio = (val - min) / (max - min);

	// RGB
	// 111 = white
	// 000 = black
	// 011 = cyan
	// 101 = magenta
	// 110 = yellow

  if(ratio > 0.0 && ratio < (0.2))
  {
    interpolate( (ratio - 0.0) / 0.2,   0,   0,   0, 
                                        0,   0, 255, r, g, b);
  }
  else if(ratio > (0.2) && ratio < (0.4))
  {
    interpolate( (ratio - 0.2) / 0.2,   0,   0, 255, 
                                        0, 255, 255, r, g, b);
  }
  else if(ratio > (0.4) && ratio < (0.6))
  {
    interpolate( (ratio - 0.4) / 0.2,   0, 255, 255, 
                                        0, 255,   0, r, g, b);
  }
  else if(ratio > (0.6) && ratio < (0.8))
  {
    interpolate( (ratio - 0.6) / 0.2,   0, 255,  0,  
                                      255, 255,  0, r, g, b);
  }
  else if(ratio > (0.8) && ratio < (1.0))
  {
    interpolate( (ratio - 0.8) / 0.2,  255, 255,  0,
                                       255,   0,  0, r, g, b);
  }
  else if(ratio <= 0)
  {
    r = 0;
    g = 0;
    b = 255;
  }
  else if(ratio >= 1.0)
  {
    r = 255;
    g = 0;
    b = 0;
  }
}

inline void printInterfaceMessage()
{
  printf("Graphic Interface Manual\n");
  printf("[1]: Turn Off every color map\n");
  printf("[2]: Turn On / Off BinDensity\n");
  printf("[3]: Turn On / Off SpectralPotential Heatmap (Orange)\n");
  printf("[4]: Turn On / Off SpectralDensityGrad\n");
  printf("[5]: Turn On / Off Rows\n");
  printf("[6]: Turn On / Off Nets\n");
  printf("[7]: Turn On / Off Cluster\n");
  printf("[Q]: Close the window\n");
  printf("[Z]: Zoom In\n");
  printf("[X]: Zoom Out\n");
  printf("[F]: Zoom to Fit\n");
  printf("[H]: Print Key Map\n");
  printf("[UP DOWN LEFT DOWN]: Move Zoom Box\n");
}

void
Painter::init()
{
  numArrow_ = 800;
  clusterHighlight_ = true;

  // In some Bookshelf benchmarks like adaptec1,
  // some macro blocks are outside die so that they cannot
  // captured by ux uy of the die
  // We have to visit every cells to get the 'true' Ux Uy
  float dieUx = db_->die()->ux();
  float dieUy = db_->die()->uy();
  float dieLx = db_->die()->lx();
  float dieLy = db_->die()->ly();

  float dieDx = dieUx - dieLx;
  float dieDy = dieUy - dieLy;

  maxWidth_  = db_->die()->ux();
  maxHeight_ = db_->die()->uy();

  for(auto c : db_->cells())
  {
    if(c->ux() > maxWidth_)
      maxWidth_ = c->ux();
    if(c->uy() > maxHeight_)
      maxHeight_ = c->uy();
  }

  lenIO_ = std::max(dieDx, dieDy) / 40.0;

  double scaleX = double(MAX_W) / double(maxWidth_ );
  double scaleY = double(MAX_H) / double(maxHeight_);

  double scaleXgif = double(MAX_H_PLOT) / double(maxWidth_);
  double scaleYgif = double(MAX_H_PLOT) / double(maxHeight_);

  scale_        = std::min(scaleX,    scaleY   );
  scale_window_ = std::min(scaleX,    scaleY   );
  scale_gif_    = std::min(scaleXgif, scaleYgif);

  //offsetX_ = DIE_OFFSET_X + static_cast<int>(lenIO_ * scale_);
  //offsetY_ = DIE_OFFSET_Y + static_cast<int>(lenIO_ * scale_);

  offsetX_ = static_cast<int>(MAX_W / 2.0) 
           - static_cast<int>(dieDx  * scale_ / 2.0) 
           + static_cast<int>(lenIO_ * scale_);

  offsetY_ = static_cast<int>(MAX_H / 2.0) 
           - static_cast<int>(dieDy  * scale_ / 2.0) 
           + static_cast<int>(lenIO_ * scale_);

  offsetXgif_ = static_cast<int>(MAX_W_PLOT / 2.0) 
              - static_cast<int>(dieDx  * scale_gif_ / 2.0) 
              + static_cast<int>(lenIO_ * scale_gif_);

  offsetYgif_ = static_cast<int>(MAX_H_PLOT / 2.0) 
              - static_cast<int>(dieDy  * scale_gif_ / 2.0) 
              + static_cast<int>(lenIO_ * scale_gif_);

  canvasX_ = MAX_W + 2 * offsetX_;
  canvasY_ = MAX_H + 2 * offsetY_;

  canvasX_plot_ = MAX_W_PLOT + 2 * offsetXgif_;
  canvasY_plot_ = MAX_H_PLOT + 2 * offsetYgif_;

  // canvas is just a background image for placement visualization
  canvas_ = new CImg<unsigned char>(canvasX_, canvasY_, 1, 3, 255);
  canvas_->draw_rectangle(0, 0, canvasX_, canvasY_, BACKGROND_COLOR);

  canvas_plot_ = new CImg<unsigned char>(canvasX_plot_, canvasY_plot_, 1, 3, 255);
  canvas_plot_->draw_rectangle(0, 0, canvasX_plot_, canvasY_plot_, BACKGROND_COLOR);

  // img_ := Original image which represents the whole placement
  // any 'zoomed' image will use a crop of this img_
  img_ = new CImg<unsigned char>(*canvas_);
}

int
Painter::getX(int dbX)
{
  double tempX = static_cast<double>(dbX);
  tempX = scale_ * tempX;
  return (static_cast<double>(tempX) + offsetX_);
}

int
Painter::getY(int dbY)
{
  double tempY = static_cast<double>(maxHeight_ - dbY);
  tempY = scale_ * tempY;
  return (static_cast<int>(tempY) + offsetY_);
}

int
Painter::getX(float dbX)
{
  double tempX = static_cast<double>(dbX);
  tempX = scale_ * tempX;
  return (static_cast<double>(tempX) + offsetX_);
}

int
Painter::getY(float dbY)
{
  double tempY = static_cast<double>(maxHeight_ - dbY);
  tempY = scale_ * tempY;
  return (static_cast<int>(tempY) + offsetY_);
}

void 
Painter::drawLine(CImgObj *img, 
                  int x1, 
                  int y1, 
                  int x2, 
                  int y2)
{
  img->draw_line(x1, y1, x2, y2, black);
}

void 
Painter::drawLine(CImgObj *img, int x1, int y1, int x2, int y2, Color color)
{
  img->draw_line(x1, y1, x2, y2, color);
}

void
Painter::drawRect(CImgObj *img, int lx, int ly, int ux, int uy, Color rect_c, int w)
{
  drawRect(img, lx, ly, ux, uy, rect_c, black, w, 1.0);
}

void
Painter::drawRect(CImgObj *img, int lx, int ly, int ux, int uy, Color rect_c,
                  Color line_c, int w, float opacity)
{
  img->draw_rectangle(lx, ly, ux, uy, rect_c, opacity);
  drawLine(img, lx, ly, ux, ly, line_c);
  drawLine(img, ux, ly, ux, uy, line_c);
  drawLine(img, ux, uy, lx, uy, line_c);
  drawLine(img, lx, uy, lx, ly, line_c);

  int xd = (ux > lx) ? 1 : -1; 
  int yd = (uy > ly) ? 1 : -1; 

  if(w > 0)
  {
    for(int i = 1; i < w + 1; i++)
    {
      drawLine(img, 
               lx + xd * i, ly + yd * i, 
               ux - xd * i, ly + yd * i, line_c);

      drawLine(img,
               ux - xd * i, ly + yd * i, 
               ux - xd * i, uy - yd * i, line_c);

      drawLine(img,
               ux - xd * i, uy - yd * i, 
               lx + xd * i, uy - yd * i, line_c);

      drawLine(img,
               lx + xd * i, uy - yd * i, 
               lx + xd * i, ly + yd * i, line_c);

      drawLine(img,
               lx - xd * i, ly - yd * i, 
               ux + xd * i, ly - yd * i, line_c);

      drawLine(img,
               ux + xd * i, ly - yd * i, 
               ux + xd * i, uy + yd * i, line_c);

      drawLine(img,
               ux + xd * i, uy + yd * i, 
               lx - xd * i, uy + yd * i, line_c);

      drawLine(img,
               lx - xd * i, uy + yd * i, 
               lx - xd * i, ly - yd * i, line_c);
    }
  }
}

bool 
Painter::check_inside(int lx, int ly, int w, int h)
{
  if(lx < 0)            return false;
  if(ly < 0)            return false;
  if(lx + w > canvasX_) return false;
  if(ly + h > canvasY_) return false;
  return true;
}

void
Painter::drawDie(CImgObj *img, const Die* die)
{
  drawRect(img, getX(die->lx()), getY(die->ly()), 
                getX(die->ux()), getY(die->uy()), 
                DIE_COLOR, DIE_LINE_COLOR, DIE_LINE_THICKNESS, DIE_OPACITY);
}

void
Painter::drawRow(CImgObj *img, const Row* row)
{
  int rowLx = getX(row->lx());
  int rowLy = getY(row->ly());
  int rowUx = getX(row->ux());
  int rowUy = getY(row->uy());

  drawLine(img, rowLx, rowLy, rowUx, rowLy, palegreen);
  drawLine(img, rowUx, rowLy, rowUx, rowUy, palegreen);
  drawLine(img, rowUx, rowUy, rowLx, rowUy, palegreen);
  drawLine(img, rowLx, rowUy, rowLx, rowLy, palegreen);

  float xPt   = row->lx();
  float stepX = row->stepX();

  int numSiteX = row->numSiteX();
  if(numSiteX < 4000)
  {
    for(int i = 1; i < row->numSiteX(); i++)
    {
      xPt += stepX;
      int newX = getX(xPt);
      drawLine(img, newX, rowLy, newX, rowUy, palegreen);
    }
  }
}

void
Painter::drawRows(CImgObj* img)
{
  int numSiteMax = 0;
  for(auto& row : db_->rows())
  {
    int numSite = row->numSiteX();
    if(numSite > numSiteMax)
      numSiteMax = numSite;
    drawRow(img, row);
  }

  if(numSiteMax > 4000)
    printf("Warn - Too many Sites. Only boudary will be visualized.\n");
}

void
Painter::drawNet(CImgObj *img, const Net* net)
{
//  int cx = getX(net->cx());
//  int cy = getY(net->cy());
//
//  // terminal
//  int tx, ty = 0;
//
//  // Draw Start for a hyper edge
//  for(auto& pin : net->pins())
//  {
//    tx = getX(pin->cx());
//    ty = getY(pin->cy());
//    drawLine(img, cx, cy, tx, ty, darkblue);
//  }

  // Draw Start for a hyper edge
  int numLine = 0;
  for(int pinID1 = 0; pinID1 < net->deg() - 1; pinID1++)
  {
    if(!net->pins()[pinID1]->cell()->isMacro() && !net->pins()[pinID1]->isIO())
      continue;

    int x1 = getX(net->pins()[pinID1]->cx());
    int y1 = getY(net->pins()[pinID1]->cy());

    for(int pinID2 = pinID1 + 1; pinID2 < net->deg(); pinID2++)
    {
      if(!net->pins()[pinID2]->cell()->isMacro() && !net->pins()[pinID2]->isIO())
        continue;

      if(net->pins()[pinID1]->cell() == net->pins()[pinID2]->cell())
        continue;

      int x2 = getX(net->pins()[pinID2]->cx());
      int y2 = getY(net->pins()[pinID2]->cy());

      if(net->pins()[pinID2]->cell()->isMacro() && net->pins()[pinID1]->cell()->isMacro())
        drawLine(img, x1, y1, x2, y2, green);
      else
        drawLine(img, x1, y1, x2, y2, darkblue);
      //std::cout << "Draw " << ++numLine << " lines..." << std::endl;
    }
  }
}

void
Painter::drawNets(CImgObj* img)
{
  int num_macro = 0;
  int num_io    = 0;

  for(auto& net : db_->nets())
  {
    if(net->deg() > 1000)
      continue;
    for(auto& pin : net->pins())
    {
      if(pin->cell()->isMacro()) 
        num_macro++;
      if(pin->cell() == nullptr)
        num_io++;
    }

    if(num_macro >= 2 || ( (num_io >= 1) && num_macro >= 1 ) )
      drawNet(img, net);

    num_macro = 0;
    num_io    = 0;
  }
}

void
Painter::drawCell(CImgObj* img, const Cell* cell)
{
  int newLx = getX(cell->lx());
  int newLy = getY(cell->ly());
  int newUx = getX(cell->ux());
  int newUy = getY(cell->uy());

  if(cell->isMacro())
  {
    drawRect(img, newLx, newLy, newUx, newUy, MACRO_COLOR, 
                                              MACRO_LINE_COLOR,
                                              MACRO_LINE_THICKNESS, 
                                              MACRO_OPACITY);

    if(cell->dx() > db_->die()->dx() * 0.2 ||  cell->isMacro() )
    {
      //if(db_->ifBookShelf())
      //  img->draw_text(newLx, newUy, cell->bsCellPtr()->name().c_str(), white, NULL, 1, 40); // 80 for show()?
    }
  }
  else 
  {
    drawRect(img, newLx, newLy, newUx, newUy, STD_CELL_COLOR, 
                                              STD_CELL_LINE_COLOR, 
                                              STD_CELL_LINE_THICKNESS,
                                              STD_CELL_OPACITY);
  }
}

void
Painter::drawFixed(CImgObj *img, const Cell* cell)
{
  int newLx = getX(cell->lx());
  int newLy = getY(cell->ly());
  int newUx = getX(cell->ux());
  int newUy = getY(cell->uy());

  if(cell->isFixed())
  {
    drawRect(img, newLx, newLy, newUx, newUy, FIXED_CELL_COLOR, 
                                              MACRO_LINE_COLOR,
                                              MACRO_LINE_THICKNESS, 
                                              FIXED_CELL_OPACITY);
  }
}

void
Painter::drawFiller(CImgObj *img, const Cell* cell)
{
  int newLx = getX(cell->lx());
  int newLy = getY(cell->ly());
  int newUx = getX(cell->ux());
  int newUy = getY(cell->uy());

  drawRect(img, newLx, newLy, newUx, newUy, FILLER_CELL_COLOR, 
                                            FILLER_LINE_COLOR, 
                                            STD_CELL_LINE_THICKNESS,
                                            FILLER_CELL_OPACITY);
}

void
Painter::drawIO(CImgObj *img, const Cell* cell, 
                float dieLx, 
                float dieLy, 
                float dieUx,
                float dieUy)
{
  float ioLx = cell->lx();
  float ioLy = cell->ly();
  float ioUx = cell->ux();
  float ioUy = cell->uy();

  float p1X, p1Y, p2X, p2Y, p3X, p3Y = 0;

  // 0.866 ~= sqrt(3)/2
  // Case #1: IO is below Die
  if(ioLy <= dieLy)
  {
    p1X = (ioLx + ioUx) / 2;
    p1Y = dieLy;
    p2X = p1X - lenIO_ / 2;
    p2Y = p1Y - 0.866 * lenIO_;
    p3X = p1X + lenIO_ / 2;
    p3Y = p1Y - 0.866 * lenIO_;
  }
  // Case #2: IO is above Die
  else if(ioUy >= dieUy)
  {
    p1X = (ioLx + ioUx) / 2;
    p1Y = dieUy;
    p2X = p1X - lenIO_ / 2;
    p2Y = p1Y + 0.866 * lenIO_;
    p3X = p1X + lenIO_ / 2;
    p3Y = p1Y + 0.866 * lenIO_;
  }
  // Case #3: IO is left to Die
  else if(ioLx <= dieLx)
  {
    p1X = dieLx;
    p1Y = (ioUy + ioLy) / 2;
    p2X = p1X - 0.866 * lenIO_;
    p2Y = p1Y + lenIO_ / 2;
    p3X = p1X - 0.866 * lenIO_;
    p3Y = p1Y - lenIO_ / 2;
  }
  // Case #4: IO is right to Die
  else if(ioUx >= dieUx)
  {
    p1X = dieUx;
    p1Y = (ioUy + ioLy) / 2;
    p2X = p1X + 0.866 * lenIO_;
    p2Y = p1Y + lenIO_ / 2;
    p3X = p1X + 0.866 * lenIO_;
    p3Y = p1Y - lenIO_ / 2;
  }
  else
  {
    //printf("IO  (%10.0f, %10.0f) - (%10.0f, %10.0f)\n",  ioLx,  ioLy,  ioUx,  ioUy);
    //printf("Die (%10.0f, %10.0f) - (%10.0f, %10.0f)\n", dieLx, dieLy, dieUx, dieUy);
    p1X = ioLx - 0.5   * (lenIO_ * 0.1);
    p1Y = ioLy - 0.333 * (lenIO_ * 0.1);
    p2X = ioLx + 0.5   * (lenIO_ * 0.1);
    p2Y = ioLy - 0.333 * (lenIO_ * 0.1);
    p3X = ioLx;
    p3Y = ioLy + 0.666 * (lenIO_ * 0.1);
    //return;
  }

  int p1Xi, p1Yi, p2Xi, p2Yi, p3Xi, p3Yi = 0;

  p1Xi = getX(p1X);
  p1Yi = getY(p1Y);
  p2Xi = getX(p2X);
  p2Yi = getY(p2Y);
  p3Xi = getX(p3X);
  p3Yi = getY(p3Y);

  img->draw_triangle(p1Xi, p1Yi, 
                     p2Xi, p2Yi, 
                     p3Xi, p3Yi,
                     IO_COLOR, 
                     IO_OPACITY);
}

void
Painter::drawCluster(CImgObj* img, const Cell* cell)
{
  int newLx = getX(cell->lx());
  int newLy = getY(cell->ly());
  int newUx = getX(cell->ux());
  int newUy = getY(cell->uy());

  Color color;
	Color line_color;

  // TODO: Fix this stupid code
  if(color_map_.size() == 0 || cell->clusterID() == db_->numCluster())
    color = STD_CELL_COLOR; 
  else
    color = color_map_[cell->clusterID()];

	if(cell->isMacro())
		line_color = black;
	else
		line_color = color;

  drawRect(img, newLx, newLy, newUx, newUy, color, 
                                            line_color, 
                                            STD_CELL_LINE_THICKNESS,
                                            1.0);
}

void
Painter::drawCells(CImgObj *img, 
                   const std::vector<Cell*>& cells, 
                   bool filler)
{
  float dieLx = db_->die()->lx();
  float dieLy = db_->die()->ly();
  float dieUx = db_->die()->ux();
  float dieUy = db_->die()->uy();

  int numIODraw = 0;

  for(auto &c : cells)
  {
    if( c->isFixed() )
    {
//      if( c->isIO() )
//			{
//        drawIO(img, c, dieLx, dieLy, dieUx, dieUy);
//			}
//      else
        drawFixed(img, c);
    }
    else if( c->isFiller() )
    {
      if(filler) 
        drawFiller(img, c);
    }
    else
    {
      if(clusterHighlight_)
        drawCluster(img, c);
      else
        drawCell(img, c);
    }
  }
}

void
Painter::genColorMap()
{
  int numCluster = db_->numCluster();

  if(numCluster == 0)
    clusterHighlight_ = false;

  for(int i = 0 ; i < numCluster; i++)
  {
		srand(i);

    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;

    // std::cout << "Cluster " << i << " -> " << r << " " << g << " " << b << std::endl;

    unsigned char* color = new unsigned char[3];
    color[0] = static_cast<unsigned char>(r); 
    color[1] = static_cast<unsigned char>(g); 
    color[2] = static_cast<unsigned char>(b); 

    // TODO: Make delete
    color_map_.push_back(color);
  }
}

void
Painter::drawChip()
{
  if(clusterHighlight_ && color_map_.size() == 0)
    genColorMap();

  // Since getX and getY uses
  // scale_, offsetX_, offsetY_ by default
  // we have to chage these variables
  // in the plotMode / guiMode, respectively
  scale_   = scale_window_;
  offsetX_ = offsetX_;
  offsetY_ = offsetY_;

  drawDie(img_, db_->die() );
  drawCells(img_, db_->cells(), true);

  show();
}

void 
Painter::show()
{
  int viewX = 0;
  int viewY = 0;

  int ZoomBoxW = MAX_H; 
  int ZoomBoxH = MAX_W; 

  bool redraw = false;

  bool heatmapSpectral      = false;
  bool heatmapBinDensity    = false;
  bool densityGradSpectral  = false;
  bool showRows             = false;
  bool showNets             = false;

  int lx = 0;
  int ly = 0;

  CImgObj ZoomBox 
    = img_->get_crop(lx, ly, lx + canvasX_, ly + canvasY_);

  window_ = new CImgDisplay(WINDOW_W, WINDOW_H, "Placement Plot");

  printInterfaceMessage();

  // Interactive Mode //
  while(!window_->is_closed() && !window_->is_keyESC()
                              && !window_->is_keyQ())
  {
    if(redraw)
    {
      img_ = new CImg<unsigned char>(*canvas_);
      drawDie(img_, db_->die() );
      drawCells(img_, db_->cells() );

      if(heatmapSpectral)
        drawBinsSpectralPotential(img_);
      if(heatmapBinDensity)
        drawBinsDensity(img_);
      if(densityGradSpectral)
        drawDensityGradSpectral(img_);
      if(showRows)
        drawRows(img_);
      if(showNets)
        drawNets(img_);

      ZoomBox 
        = img_->get_crop(lx, ly, lx + ZoomBoxW, ly + ZoomBoxH);
      ZoomBox.resize(*window_);
      redraw = false;
    }

    ZoomBox.display(*window_);

    if(window_->key())
    {
      switch(window_->key())
      {
        case cimg::keyARROWUP:
          if(check_inside(lx, ly - MOVE_SPEED, ZoomBoxW, ZoomBoxH))
          {
            ly -= MOVE_SPEED;
            redraw = true;
          }
          break;
        case cimg::keyARROWDOWN:
          if(check_inside(lx, ly + MOVE_SPEED, ZoomBoxW, ZoomBoxH))
          {
            ly += MOVE_SPEED;
            redraw = true;
          }
          break;
        case cimg::keyARROWLEFT:
          if(check_inside(lx - MOVE_SPEED, ly, ZoomBoxW, ZoomBoxH))
          {
            lx -= MOVE_SPEED;
            redraw = true;
          }
          break;
        case cimg::keyARROWRIGHT:
          if(check_inside(lx + MOVE_SPEED, ly, ZoomBoxW, ZoomBoxH))
          {
            lx += MOVE_SPEED;
            redraw = true;
          }
          break;

        case cimg::keyZ:
          if(ZoomBoxW > ZOOM_SPEED 
          && ZoomBoxH > ZOOM_SPEED)
          {
            redraw = true;
            ZoomBoxW -= ZOOM_SPEED;
            ZoomBoxH -= ZOOM_SPEED;
          }
          break;
        case cimg::keyX:
          if(ZoomBoxW <= canvasX_ - ZOOM_SPEED 
          && ZoomBoxH <= canvasY_ - ZOOM_SPEED)
          {
            redraw = true;
            ZoomBoxW += ZOOM_SPEED;
            ZoomBoxH += ZOOM_SPEED;
          }
          break;

        case cimg::keyF:
          redraw = true;
          lx = 0;
          ly = 0;
          ZoomBoxW = MAX_H; 
          ZoomBoxH = MAX_W; 
          break;

        case cimg::keyH:
          printInterfaceMessage();
          break;

        case cimg::key1:
          redraw = true;
          heatmapSpectral     = false;
          heatmapBinDensity   = false;
          heatmapSpectral     = false;
          showRows            = false;
          break;

        case cimg::key2:
          redraw = true;
          heatmapSpectral   = false;
          heatmapBinDensity = !heatmapBinDensity;
          break;

        case cimg::key3:
          redraw = true;
          heatmapSpectral   = !heatmapSpectral;
          heatmapBinDensity = false;
          break;

        case cimg::key4:
          redraw = true;
          densityGradSpectral = !densityGradSpectral;
          break;

        case cimg::key5:
          redraw = true;
          showRows = !showRows;
          break;

        case cimg::key6:
          redraw = true;
          showNets = !showNets;
          break;

        case cimg::key7:
          redraw = true;
          clusterHighlight_ = !clusterHighlight_;
          break;
      }
      window_->set_key(); // Flush all key events...
    }
    window_->wait();
  }
  exit(0);
}

void
Painter::drawBin(CImgObj *img, 
                 const Bin* bin, 
                 float value,
                 float max, 
                 float min)
{
  int newLx = getX(bin->lx());
  int newLy = getY(bin->ly());
  int newUx = getX(bin->ux());
  int newUy = getY(bin->uy());

  int Cx = (newLx + newUx) / 2;
  int Cy = (newLy + newUy) / 2;

  int r, g, b = 0;
  setRGB(r, g, b, value, max, min);

  const unsigned char color[] = {static_cast<unsigned char>(r), 
                                 static_cast<unsigned char>(g), 
                                 static_cast<unsigned char>(b)};

  // For heatmap
  img->draw_rectangle(newLx, newLy, newUx, newUy, color, 0.5);
}

void
Painter::drawBinsSpectralPotential(CImgObj *img)
{
  // printf("[Plot] Drawing Bins (Spectral Heatmap).\n");

  float maxP = FLT_MIN;
  float minP = FLT_MAX;

  float sumPoten = 0.0;

  for(auto &bin : db_->bins())
  {
    float p = bin->potential();
    if(p > maxP)
      maxP = p;
    if(p < minP)
      minP = p;

    sumPoten += p;
  }

//  printf("MaxPotential is %E\n", maxP);
//  printf("MinPotential is %E\n", minP);
//  printf("SumPotential is %E\n", sumPoten);
//  printf("AvgPotential is %E\n", sumPoten / db_->bins().size());

  for(auto &bin : db_->bins())
    drawBin(img, bin, bin->potential(), maxP, minP);
}

void
Painter::drawBinsDensity(CImgObj *img)
{
  // printf("[Plot] Drawing Bins (BinDensity Heatmap).\n");

  float maxD = FLT_MIN;
  float minD = FLT_MAX;

  float sumDensity = 0.0;

  for(auto &bin : db_->bins())
  {
    float d = bin->density();
    if(d > maxD)
      maxD = d;
    if(d < minD)
      minD = d;
    sumDensity += d;
  }

  for(auto &bin : db_->bins())
    drawBin(img, bin, bin->density(), maxD, minD);
}

void
Painter::drawDensityGradSpectral(CImgObj *img)
{
  // printf("[Plot] Drawing DensityGrad (Spectral).\n");

  int numCell = db_->numMovable();

  float sizeS    = 0.0;
  float maxSizeS = 0.0;
  float avgSizeS = 0.0;
  float sumSizeS = 0.0;

  for(int i = 0; i < numCell; i++)
  {
    float xS = db_->densityGradX()[i];
    float yS = db_->densityGradY()[i];

    sizeS = std::sqrt(xS * xS + yS * yS);
    sumSizeS += sizeS;

    if(sizeS > maxSizeS)
      maxSizeS = sizeS;
  }

  avgSizeS = sumSizeS / numCell;

  // printf("[Plot] MaxSizeSpectral : %E\n", maxSizeS);
  // printf("[Plot] AvgSizeSpectral : %E\n", avgSizeS);

  float refSize = 100.0 * db_->die()->dx(); // 0.4?
  float sizeForSpectral = refSize / maxSizeS;

  // printf("[Plot] SizeForSpectral : %E\n", sizeForSpectral);

  for(int cellID = 0; cellID < numCell; cellID++)
  {
    if(cellID % numArrow_ != 0)
      continue;

    int cX = db_->movableCells()[cellID]->cx();
    int cY = db_->movableCells()[cellID]->cy();

    int startX = getX(cX);
    int startY = getY(cY);

    float scaledExS = db_->densityGradX()[cellID] * sizeForSpectral;
    float scaledEyS = db_->densityGradY()[cellID] * sizeForSpectral;

    int endXS = getX(cX + scaledExS);
    int endYS = getY(cY + scaledEyS);

    img->draw_arrow(startX, startY, endXS, endYS, orange);
  }
}

void
Painter::makeDir()
{
  std::string benchName = db_->designName();
  std::string benchDir  = db_->designDir();

  benchDir += "/";

  outputDir_ = benchDir + "cell_plot/";

  std::string command = "mkdir -p " + outputDir_;

  std::system(command.c_str());
}

void
Painter::prepareForPlot()
{
  makeDir();

  if(clusterHighlight_)
    genColorMap();
}

void
Painter::saveImage(int iter, float hpwl, float overflow, bool filler)
{
  scale_   = scale_gif_;
  offsetX_ = offsetXgif_;
  offsetY_ = offsetYgif_;

  CImgObj imgForThisIter(*canvas_plot_);

  drawDie(&imgForThisIter, db_->die() );
  drawCells(&imgForThisIter, db_->cells(), filler);

  // Heatmap
  //drawBinsSpectralPotential(&imgForThisIter);

  std::string info = "Iter: ";
  info += std::to_string(iter) + " HPWL: ";
  info += std::to_string(static_cast<int64_t>(hpwl)) + " Overflow: ";
  info += std::to_string(overflow); 
  //imgForThisIter.draw_text(10, 10, info.c_str(), white, NULL, 1, 40);

  std::ostringstream iterString;
  iterString << std::setw(4) << std::setfill('0') << iter;

  std::string fileName = outputDir_ + iterString.str() + ".jpeg";
  imgForThisIter.save_jpeg(fileName.c_str(), 100);

  printf("[Plot] Save Image.\n");
}

} // namespace 
