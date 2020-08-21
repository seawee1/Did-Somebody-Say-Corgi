import sys
from os.path import join as opjoin
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.Qt import Qt
from PIL import Image
import pandas as pd

class MainWindow(QMainWindow):

    def __init__(self):
        # Basic stuff
        super(MainWindow, self).__init__()
        self.title = "Annotation Tool"
        self.setWindowTitle(self.title)

        # Dataset stuff
        self.dataset_path = 'H://reddit/database_1597063213/'
        self.df = pd.read_csv(opjoin(self.dataset_path, 'submissions_.csv'))
        self.df
        self.cur_index = -1

        # Define label to display image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.mousePressEvent = self.moveBBox
        self.setCentralWidget(self.label)

        self.display_size = 500
        self.target_size = 1024
        self.resize(1024, 786)

        self.painter = None

    def prepareImage(self):
        im = Image.open(self.imagepath)

        # Scale shorter side to self.display_size while retaining aspect ratio
        self.w, self.h = im.size
        self.isBroad = (self.w > self.h)
        if self.isBroad:
            self.scale_f = self.display_size / self.h
            self.w = int(self.scale_f * self.w)
            self.h = self.display_size
        else:
            self.scale_f = self.display_size / self.w
            self.w = self.display_size
            self.h = int(self.scale_f * self.h)

    def displayImage(self):
        # Load image
        # PIL image is needed because QPixmax width() and height() cause crash on Windows
        self.pixmap = QPixmap(self.imagepath)
        im = Image.open(self.imagepath)

        # Scale shorter side to self.display_size while retaining aspect ratio
        if self.isBroad:
            self.pixmap = self.pixmap.scaledToHeight(self.display_size)
        else:
            self.pixmap = self.pixmap.scaledToWidth(self.display_size)

        # Create painter
        self.painter = QPainter(self.pixmap)

        # Configure painter
        self.penRectangle = QPen(Qt.red)
        self.penRectangle.setWidth(3)
        self.painter.setPen(self.penRectangle)

        # Draw bounding box rectangle
        x = min(max(self.xPos + self.xOffset, 0.0), self.w-self.display_size)
        y = min(max(self.yPos + self.yOffset, 0.0), self.h-self.display_size)
        self.painter.drawRect(x, y, self.display_size-1, self.display_size-1)
        self.label.setPixmap(self.pixmap)

        del self.painter

    def moveBBox(self, event):
        x = event.pos().x() - self.label.width()/2
        y = event.pos().y() - self.label.height()/2

        self.xOffset, self.yOffset = 0.0, 0.0
        if self.isBroad:
            self.xOffset = x
        else:
            self.yOffset = y

        self.displayImage()

    def keyPressEvent(self, event):
        key = event.key()
        print(key)

        new_image = False
        if event.key() == Qt.Key_D:
            new_image = True
            self.cur_index = self.cur_index + 1 if self.cur_index + 1 < len(self.df) else len(self.df) - 1
        elif event.key() == Qt.Key_A:
            new_image = True
            self.cur_index = self.cur_index - 1 if self.cur_index - 1 >= 0 else 0
        elif event.key() == Qt.Key_X:
            # TODO: Flag image
            pass

        if new_image:
            self.imagepath = opjoin(self.dataset_path, 'images', self.df.iloc[self.cur_index]['imgfile'])

            self.prepareImage()
            self.xPos = (0.5 * self.w - 0.5 * self.display_size)
            self.yPos = (0.5 * self.h - 0.5 * self.display_size)
            self.xOffset, self.yOffset = 0.0, 0.0
            self.displayImage()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())