import sys
import os
from os.path import join as opjoin

from PIL import Image
import pandas as pd
import numpy as np
import pathlib

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QAction, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QKeyEvent
from PyQt5.Qt import Qt, QEvent

class MainWindow(QMainWindow):

    def __init__(self):
        # Basic stuff
        super(MainWindow, self).__init__()
        self.title = "Annotation Tool"
        self.setWindowTitle(self.title)

        # Shorter side of image is scaled to this size
        self.displaySize = 500
        self.targetSize = 1024
        self.bboxSteps = 5
        self.resize(1200, 800)
        self.setFixedSize(1200, 800)


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

        self.show()

    def openDataset(self):
        # Open dataset .csv
        file_name = QFileDialog.getOpenFileName(self, 'Open File')
        self.dataset_path = os.path.dirname(file_name[0])
        self.df = pd.read_csv(filename[0])
        #pathlib.Path(self.dataset_path, 'labels').mkdir(parents=True, exist_ok=True)

        # Add flagged and crop_offset columns
        #self.df['flagged'] = False
        #self.df['bboxX'] = np.nan
        #self.df['bboxY'] = np.nan
        #self.df['bboxSize'] = np.nan

        # Display first image
        self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_F, Qt.NoModifier))

    def saveDataset(self):
        # Save dataset to .csv
        filename = QFileDialog.getSaveFileName(self, 'Save File')
        with open(filename[0], 'w', encoding='utf-8') as f:
            self.df.to_csv(f, index = False, line_terminator='\n')

    def prepareImage(self):
        self.xOffset, self.yOffset, self.bboxScale = 0.0, 0.0, 1.0
        # Load image
        self.im = Image.open(self.imagepath)
        self.pixmap = QPixmap(self.imagepath)

        # Scale shorter side to self.display_size while retaining aspect ratio
        self.w, self.h = self.im.size
        self.isBroad = (self.w > self.h)
        if self.isBroad:
            self.bboxMax = self.h

            self.scale_f = self.displaySize / self.h
            self.w = int(self.scale_f * self.w)
            self.h = self.displaySize
        else:
            self.bboxMax = self.w

            self.scale_f = self.displaySize / self.w
            self.w = self.displaySize
            self.h = int(self.scale_f * self.h)

        self.bboxMin = self.targetSize if self.targetSize < self.bboxMax else self.bboxMax
        self.bboxStep = (self.bboxMax - self.bboxMin) / self.bboxSteps

        if self.isBroad:
            self.pixmap = self.pixmap.scaledToHeight(self.displaySize)
        else:
            self.pixmap = self.pixmap.scaledToWidth(self.displaySize)


    # xOffset and yOffset are for crop bounding box
    def displayImage(self):
        pixmap = self.pixmap.copy()
        # Create painter
        self.painter = QPainter(pixmap)

        # Configure painter
        self.penRectangle = QPen(Qt.red)
        self.penRectangle.setWidth(3)
        self.painter.setPen(self.penRectangle)

        # Initial position of bounding box
        bboxSize = self.bboxMin * self.scale_f  +  self.bboxScale*self.bboxSteps*self.bboxStep*self.scale_f
        xPos = (0.5 * self.w - 0.5 * bboxSize)
        yPos = (0.5 * self.h - 0.5 * bboxSize)

        # Add offset
        x = min(max(xPos + self.xOffset, 0.0), self.w-bboxSize)
        y = min(max(yPos + self.yOffset, 0.0), self.h-bboxSize)

        # Draw bounding box rectangle and center point
        self.painter.drawRect(x, y, bboxSize-1, bboxSize-1)
        self.painter.drawPoint(x+(bboxSize-1)/2, y+(bboxSize-1)/2)
        self.label.setPixmap(pixmap)

        # Save crop_offset to dataframe
        scale_f_inv = 1.0/self.scale_f
        bboxX, bboxY, bboxSize_ = int(x * scale_f_inv), int(y * scale_f_inv), int(bboxSize * scale_f_inv)
        print('{:d}/{:d} | X: {:d}, Y: {:d}, S: {:d}'.format(self.cur_index, len(self.df), bboxX, bboxY, bboxSize_))
        self.df.loc[self.cur_index, 'bboxX'] = bboxX
        self.df.loc[self.cur_index, 'bboxY'] = bboxY
        self.df.loc[self.cur_index, 'bboxSize'] = bboxSize_

        del self.painter

    # Moves crop bbox on click
    def moveBBox(self, event):
        if self.df is None:
            return

        self.xOffset = event.pos().x() - self.label.width()/2
        self.yOffset = event.pos().y() - self.label.height()/2

        self.displayImage()


    def keyPressEvent(self, event):
        key = event.key()
        new_image = False
        if event.key() == Qt.Key_F: # Next image
            new_image = True
            self.cur_index = self.cur_index + 1 if self.cur_index + 1 < len(self.df) else len(self.df) - 1
        elif event.key() == Qt.Key_A: # Previous image
            new_image = True
            self.cur_index = self.cur_index - 1 if self.cur_index - 1 >= 0 else 0
        elif event.key() == Qt.Key_L: # Move BBox
            self.xOffset += 20.0
            self.displayImage()
        elif event.key() == Qt.Key_H:
            self.xOffset -= 20.0
            self.displayImage()
        elif event.key() == Qt.Key_J:
            self.yOffset += 20.0
            self.displayImage()
        elif event.key() == Qt.Key_K:
            self.yOffset -= 20.0
            self.displayImage()
        elif event.key() == Qt.Key_D: # Change Bbox scale
            self.bboxScale += 1.0/self.bboxSteps
            self.bboxScale = min(1.0, self.bboxScale)
            self.displayImage()
        elif event.key() == Qt.Key_S:
            self.bboxScale -= 1.0/self.bboxSteps
            self.bboxScale = max(0.0, self.bboxScale)
            self.displayImage()
        elif event.key() == Qt.Key_G: # Flag/Unflag
            self.df.loc[self.cur_index, 'flagged'] = not self.df.loc[self.cur_index, 'flagged']
            print('Flagged:', self.df.loc[self.cur_index, 'flagged'])

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

            try:
                im = Image.open(self.imagepath)
            except:
                print('Skipping image, because cannot open')
                self.df.loc[self.cur_index, 'flagged'] = True
                self.keyPressEvent(event)


            self.prepareImage()
            self.displayImage()

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

sys.excepthook = except_hook

app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())