#ifndef LAYOUT_VIEW_H
#define LAYOUT_VIEW_H

#include <QGraphicsView>

namespace gui
{

class LayoutView : public QGraphicsView
{
  Q_OBJECT

  public:
    
    LayoutView(QWidget* parent = nullptr);

    void zoomFit();
    void zoomIn();
    void zoomOut();

  protected:

    void wheelEvent(QWheelEvent* event)  override;
    void paintEvent(QPaintEvent* event)  override;
    void keyPressEvent(QKeyEvent* event) override;

  private:

    bool firstShow_;

  public slots:

    void zoomIn_slot();
    void zoomOut_slot();
    void zoomFit_slot();
};

}

#endif
