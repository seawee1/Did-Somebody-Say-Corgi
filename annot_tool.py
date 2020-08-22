import sys
import os
from os.path import join as opjoin

from PIL import Image
import pandas as pd
import numpy as np

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QAction, QFileDialog
from PyQt5.QtGui import QPainter, QPen, QKeyEvent
from PyQt5.Qt import Qt, QEvent

class MainWindow(QMainWindow):

    def __init__(self):
        # Basic stuff
        super(MainWindow, self).__init__()
        self.title = "Annotation Tool"
        self.setWindowTitle(self.title)

        # Shorter side of image is scaled to this size
        self.display_size = 500
        self.resize(1024, 786)

        # Menu bar
        # Open dataset
        openAction = QAction('Open', self)
        openAction.setStatusTip('Opens a dataset...')
        openAction.triggered.connect(self.openDataset)

        # Save dataset
        saveAction = QAction('Save', self)
        saveAction.setStatusTip('Saves a dataset...')
        saveAction.triggered.connect(self.saveDataset)

        # Add to Menubar
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)

        # Dataset stuff
        self.dataset_path = None
        self.df = None
        self.cur_index = -1

        # Saved for currently displayed image
        self.imagepath = None
        self.im = None

        # Define label to display image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.mousePressEvent = self.moveBBox
        self.setCentralWidget(self.label)

        self.painter = None

    def openDataset(self):
        # Open dataset .csv
        filename = QFileDialog.getOpenFileName(self, 'Open File')
        self.dataset_path = os.path.dirname(filename[0])
        self.df = pd.read_csv(filename[0])

        # Add flagged and crop_offset columns
        self.df['flagged'] = False
        self.df['crop_offset'] = np.nan

        # Display first image
        self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_F, Qt.NoModifier))

    def saveDataset(self):
        # Save dataset to .csv
        filename = QFileDialog.getSaveFileName(self, 'Save File')
        with open(filename[0], 'w', encoding='utf-8') as f:
            self.df.to_csv(f, index = False, line_terminator='\n')

    # xOffset and yOffset are for crop bounding box
    def displayImage(self, xOffset=0.0, yOffset=0.0):
        # Load image
        self.pixmap = QPixmap(self.imagepath)

        # Scale shorter side to self.display_size while retaining aspect ratio
        w, h = self.im.size
        isBroad = (w > h)
        if isBroad:
            scale_f = self.display_size / h
            w = int(scale_f * w)
            h = self.display_size
        else:
            scale_f = self.display_size / w
            w = self.display_size
            h = int(scale_f * h)

        if isBroad:
            self.pixmap = self.pixmap.scaledToHeight(self.display_size)
        else:
            self.pixmap = self.pixmap.scaledToWidth(self.display_size)

        # Create painter
        self.painter = QPainter(self.pixmap)

        # Configure painter
        self.penRectangle = QPen(Qt.red)
        self.penRectangle.setWidth(3)
        self.painter.setPen(self.penRectangle)

        # Initial position of bounding box
        xPos = (0.5 * w - 0.5 * self.display_size)
        yPos = (0.5 * h - 0.5 * self.display_size)

        # Add offset
        x = min(max(xPos + xOffset, 0.0), w-self.display_size)
        y = min(max(yPos + yOffset, 0.0), h-self.display_size)

        # Draw bounding box rectangle and center point
        self.painter.drawRect(x, y, self.display_size-1, self.display_size-1)
        self.painter.drawPoint(x+(self.display_size-1)/2, y+(self.display_size-1)/2)
        self.label.setPixmap(self.pixmap)

        # Save crop_offset to dataframe
        if xOffset == 0.0 and yOffset == 0.0:
            self.df.loc[self.cur_index, 'crop_offset'] = np.nan
        else:
            self.df.loc[self.cur_index, 'crop_offset'] = int(x * 1.0/scale_f) if isBroad else int(y * 1.0/scale_f)

        del self.painter

    # Moves crop bbox on click
    def moveBBox(self, event):
        if self.df is None:
            return

        w, h = self.im.size
        isBroad = (w > h)

        x = event.pos().x() - self.label.width()/2
        y = event.pos().y() - self.label.height()/2

        xOffset, yOffset = 0.0, 0.0
        if isBroad:
            xOffset = x
        else:
            yOffset = y

        self.displayImage(xOffset, yOffset)

    def keyPressEvent(self, event):
        key = event.key()

        new_image = False
        if event.key() == Qt.Key_F: # Next image
            new_image = True
            self.cur_index = self.cur_index + 1 if self.cur_index + 1 < len(self.df) else len(self.df) - 1
        elif event.key() == Qt.Key_A: # Previous image
            new_image = True
            self.cur_index = self.cur_index - 1 if self.cur_index - 1 >= 0 else 0
        elif event.key() == Qt.Key_L: # Flag image
            self.df.loc[self.cur_index, 'flagged'] = True
        elif event.key() == Qt.Key_H: # Unflag image
            self.df.loc[self.cur_index, 'flagged'] = False

        # Open image and display
        if new_image:
            self.imagepath = opjoin(self.dataset_path, 'images', self.df.iloc[self.cur_index]['id'])
            if os.path.isfile(self.imagepath + '.jpeg'):
                self.imagepath += '.jpeg'
            elif os.path.isfile(self.imagepath + '.png'):
                self.imagepath += '.png'
            else:
                print('Skipping image, because not found')
                self.df.iloc[self.cur_index]['flagged'] = True
                self.keyPressEvent(event)

            self.im = Image.open(self.imagepath)
            self.displayImage()

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

sys.excepthook = except_hook

app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())