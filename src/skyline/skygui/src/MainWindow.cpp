#include <QMenu>
#include <QAction>
#include <QMenuBar>
#include <QToolBar>
#include <QListWidget>
#include <QDockWidget>
#include <QStatusBar>
#include <QDebug>
#include <QDesktopWidget>
#include <QGuiApplication>
#include <QScreen>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPainter>
#include <QColor>

#include <cassert>
#include <iostream>

#include "MainWindow.h"

#include "db/dbTech.h"
#include "db/dbDesign.h"

static void loadResources()
{
  Q_INIT_RESOURCE(resource);
}

namespace gui
{

MainWindow::MainWindow(QWidget *parent)
  : QMainWindow (parent)
{
}

MainWindow::~MainWindow()
{
}

void
MainWindow::init()
{
  assert(db_ != nullptr);

  loadResources();

  setWindowTitle("SkyPlace");

  // Scene
  layout_scene_ = new LayoutScene;
  layout_scene_->setBackgroundBrush( Qt::black );
  layout_scene_->setDatabase(db_);

  // View
  layout_view_ = new LayoutView;
  layout_view_->setScene(layout_scene_);
  setCentralWidget(layout_view_);

  // Menu Bar
  createMenu();

  // Dock
  createDock();

  // Tool Bar
  createToolBar();

  // Status Bar
  statusBar()->showMessage(tr("Ready"));

  // Window Size

  // this line is only for Qt5
  // I don't know why, but this returns merged size when using multiple monitors.
  QSize screenSize = QGuiApplication::primaryScreen()->size();

  // if-else to handle the problem above.
  QSize size = (screenSize.width() > 5000) ? screenSize * 0.4 
                                           : screenSize * 0.8;
  resize(size);

  // Draw Objects
  createItem();
}

void
MainWindow::createMenu()
{
  QMenu* menu;

  QFont font = menuBar()->font();
  font.setPointSize(12);
  menuBar()->setFont( font );

  menu = menuBar()->addMenu(tr("&File"));
  menu = menuBar()->addMenu(tr("&View"));
  menu = menuBar()->addMenu(tr("&Help"));
}

void
MainWindow::createDock()
{
  QDockWidget* dock = new QDockWidget(tr("Object"), this);
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);

  QListWidget* objectList = new QListWidget(dock);

  QStringList layerList;
  for(const auto layer : db_->getTech()->getLayers())
    layerList.append( QString(layer->name().c_str()) );
  objectList->addItems(layerList);

  dock->setWidget(objectList);
  addDockWidget(Qt::RightDockWidgetArea, dock);
}

void
MainWindow::createToolBar()
{
  QToolBar* toolBar;

  QAction* zoomIn  = new QAction(QIcon(":/zoom_in.png") , tr("Zoom In") , this);
  QAction* zoomOut = new QAction(QIcon(":/zoom_out.png"), tr("Zoom Out"), this);
  QAction* zoomFit = new QAction(QIcon(":/zoom_fit.png"), tr("Zoom Fit"), this);

  toolBar = addToolBar(tr("Tool Bar"));
  toolBar->addAction( zoomIn  );
  toolBar->setStatusTip(tr("Zoom In Layout View"));
  connect(zoomIn, SIGNAL(triggered()), layout_view_, SLOT(zoomIn_slot()));

  toolBar->addAction( zoomOut );
  toolBar->setStatusTip(tr("Zoom Out Layout View"));
  connect(zoomOut, SIGNAL(triggered()), layout_view_, SLOT(zoomOut_slot()));

  toolBar->addAction( zoomFit );
  toolBar->setStatusTip(tr("Zoom Fit Layout View"));
  connect(zoomFit, SIGNAL(triggered()), layout_view_, SLOT(zoomFit_slot()));
}

void
MainWindow::createItem()
{
  layout_scene_->createGuiDie();
  layout_scene_->createGuiRow();
  layout_scene_->createGuiInst();
  layout_scene_->createGuiIO();
  layout_scene_->createGuiNet();
  //layout_scene_->createGuiBlockage();
  layout_scene_->createGuiTrackGrid();
  layout_scene_->expandScene();
}

void
MainWindow::keyPressEvent(QKeyEvent* event)
{
  if(event->key() == Qt::Key_Q)
    close();
}

}
