import os
import sys
import json
import numpy as np
import logging
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import matplotlib.pyplot as plt

def sobel_convolution(image):
    """ *image* has to be 2D array. *kernel* has to be a 3x3 matrix. """

    image  = np.array(image, np.float64)
    result = np.array(image, np.float64)

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).T

    for idr, row in enumerate(image):
        for idc, column in enumerate(row):
            if idr == 0:
                y0, y1 = 0, idr + 1
                k_y0, k_y1 = 1, 2

            elif idr != image.shape[0] - 1:
                y0 = idr - 1
                y1 = idr + 1
                k_y0, k_y1 = 0, 2

            else:
                y0 = idr - 1
                y1 = idr
                k_y0, k_y1 = 0, 1

            if idc == 0:
                x0, x1 = 0, idc + 1
                k_x0, k_x1 = 1, 2

            elif idc != image.shape[1] - 1:
                x0 = idc - 1
                x1 = idc + 1
                k_x0, k_x1 = 0, 2

            else:
                x0 = idc - 1
                x1 = idc
                k_x0, k_x1 = 0, 1

            x = np.sum(image[y0:y1+1, x0:x1+1] * sobel_x[k_y0 : k_y1 + 1, k_x0 : k_x1 +1])
            y = np.sum(image[y0:y1+1, x0:x1+1] * sobel_y[k_y0 : k_y1 + 1, k_x0 : k_x1 +1])
            result[y0:y1+1, x0:x1+1] = np.sqrt(x**2 + y**2)

    return result


class ADT_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, 'main.ui'), self)

        self.setup = {}
        self.on_setup_changed()

        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderItem(0,QtGui.QStandardItem("File"))
        self.model.setHorizontalHeaderItem(1,QtGui.QStandardItem("Jet diameter"))

        self.actionExport.triggered.connect(self.on_export_clicked)
        self.actionExit.triggered.connect(self.close)


        self.tableView.setModel(self.model)


    def on_export_clicked(self):
        out = {}

        rows = self.model.rowCount()
        for row in range(rows):
            item_f, item_d = self.model.item(row, 0), self.model.item(row, 1)
            out[item_f.data(role = QtCore.Qt.DisplayRole)] = item_d.data(role = QtCore.Qt.DisplayRole)

        print(out)

        file = QtWidgets.QFileDialog.getSaveFileName(self,"Export diameter data","","JSON files (*.json)")

        with open(file[0], 'w') as f:
            f.write(json.dumps(out,  indent=4))






    def on_browse_clicked(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, ("Select Output Folder"), QtCore.QDir.currentPath())
        self.lineEdit_6.setText(folder)
        self.on_image_diretory_changed()


    def on_image_diretory_changed(self):
        """ This will update the model """

        try:
            self.setup['directory'] = self.lineEdit_6.text()
            files = [f for f in sorted(os.listdir(self.setup['directory'])) if self.setup['file_suffix'] in f]
            print(files)
        except Exception as e:
            print("no valid directory: {}".format(e))



        try:
            self.model.clear()
            self.model.setHorizontalHeaderItem(0,QtGui.QStandardItem("File"))
            self.model.setHorizontalHeaderItem(1,QtGui.QStandardItem("Jet diameter"))

            for file in files:
                item_f = QtGui.QStandardItem(file)
                item_f.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)


                item_d = QtGui.QStandardItem()
                item_d.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

                self.model.invisibleRootItem().appendRow([item_f, item_d])

        except Exception as e:
            print('Updating model failed: {}'.format(e))


    def on_run_clicked(self):
        selected_indexes = self.tableView.selectedIndexes()

        rows = self.model.rowCount()
        plt.close()
        for row in range(rows):
            self.progressBar.setValue(round(row/rows* 100.) )
            item_f, item_d = self.model.item(row, 0), self.model.item(row, 1)

            if not self.model.indexFromItem(item_f) in selected_indexes:
                continue

            file = item_f.data(role = QtCore.Qt.DisplayRole)
            file_full = os.path.join(self.setup['directory'], file)

            pos, wi = self.setup['slice_position'], self.setup['slice_width']
            ec = self.setup['edge_crop']

            sl = np.s_[:, int(pos - wi/2) : int(pos + wi/2)]

            img = sobel_convolution(plt.imread(file_full)[sl])[ec:-ec, ec:-ec]

            # plt.imshow(img)
            # plt.show()

            if self.setup['transpose_image']:
                img = img.T

            profile =  img.sum(axis = 0)

            plt.plot(np.arange(len(profile)) * self.setup['units_per_pixel'], profile, label = file)


            # Determine jet width
            c = np.ma.masked_array(profile)
            i0 = np.argmax(c)

            c = np.ma.masked_array(profile)
            mask = mask = np.array([False] * len(c))
            mask[i0-5:i0+5] = True
            c.mask = mask

            i1 = np.argmax(c)

            width = float(np.abs(i1 - i0))

            item_d.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled  | QtCore.Qt.ItemIsEditable)
            item_d.setData(self.setup['units_per_pixel'] * width, role = QtCore.Qt.DisplayRole)
            item_d.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

            app.processEvents()

        self.progressBar.setValue(100.)
        plt.show()
        plt.legend()





    def on_setup_changed(self):
        try:
            self.setup['slice_position'] = int(self.slicePositionLineEdit.text())
            self.setup['slice_width'] = int(self.sliceWidthLineEdit.text())
            self.setup['image_width'] = int(self.imageWidthPxLineEdit.text())
            self.setup['image_height'] = int(self.imageHeightPxLineEdit.text())
            self.setup['units_per_pixel'] = float(self.unitsPerPixelLineEdit.text())
            self.setup['transpose_image'] = self.transposeImageCheckBox.isChecked() == True
            self.setup['edge_crop'] = int(self.edgeCropPxLineEdit.text())
            self.setup['file_suffix'] = self.fileSuffixLineEdit.text()


        except Exception as e:
            print("Invalid setup state: {}".format(e))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ADT_MainWindow()
    window.show()
    sys.exit(app.exec_())
