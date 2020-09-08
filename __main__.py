import os
import sys
import json
import io
import csv
import numpy as np
import logging
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import matplotlib.pyplot as plt

def sobel_convolution(image):

    image  = np.array(image, np.float64)
    result = np.zeros(image.shape, np.float64)

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

            check = [x0, x1, y0, y1, k_x0, k_x1, k_y0, k_y1]

            for c in check:
                if c < 0:
                    print("negative index!")

            for k_1 in [k_x1, k_y1]:
                if k_1 > 2:
                    print("k1 too big")

            for k_0 in [k_x0, k_y0]:
                if k_1 < 0:
                    print("k0 too small")

            if image[y0:y1+1, x0:x1+1].shape != sobel_x[k_y0 : k_y1 + 1, k_x0 : k_x1 +1].shape:
                print("Kernel image missmatch")


            x = np.sum(image[y0:y1+1, x0:x1+1] * sobel_x[k_y0 : k_y1 + 1, k_x0 : k_x1 +1])
            y = np.sum(image[y0:y1+1, x0:x1+1] * sobel_y[k_y0 : k_y1 + 1, k_x0 : k_x1 +1])
            result[y0:y1+1, x0:x1+1] = y**2

    return result

def calculateCOM(e, i, window=None):

    print(window)
    if window is not None:
        e0, e1 = window
        i0, i1 = np.argmin(np.abs(e - e0)), np.argmin(np.abs(e - e1))
        i, e = i[i0:i1], e[i0:i1]

    com = 1 / np.nansum(i) * np.nansum(e * i)
    return com

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
        self.tableView.installEventFilter(self)


    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
            event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(ADT_MainWindow, self).eventFilter(source, event)


    def copySelection(self):
        selection = self.tableView.selectedIndexes()
        if selection:

            rows = self.model.rowCount()

            to_clipboard = ''
            for row in range(rows):
                item_f, item_d = self.model.item(row, 0), self.model.item(row, 1)

                if not self.model.indexFromItem(item_f) in selection:
                    continue

                name = item_f.data(role = QtCore.Qt.DisplayRole)
                diameter = item_d.data(role = QtCore.Qt.DisplayRole)

                if diameter is None:
                    diameter = ""

                to_clipboard += '{}\t{}\n'.format(name, diameter)

                app.clipboard().setText(to_clipboard)


    def on_export_clicked(self):
        try:
            out = {}

            rows = self.model.rowCount()
            for row in range(rows):
                item_f, item_d = self.model.item(row, 0), self.model.item(row, 1)
                out[item_f.data(role = QtCore.Qt.DisplayRole)] = item_d.data(role = QtCore.Qt.DisplayRole)


            file = QtWidgets.QFileDialog.getSaveFileName(self,"Export diameter data","","JSON files (*.json)")

            with open(file[0], 'w') as f:
                f.write(json.dumps(out,  indent=4))
        except Exception as e:
            print(e)




    def on_browse_clicked(self):
        try:
            folder = QtWidgets.QFileDialog.getExistingDirectory(None, ("Select Output Folder"), QtCore.QDir.currentPath())
            self.lineEdit_6.setText(folder)
            self.on_image_directory_changed()
        except Exception as e:
            print(e)



    def on_image_directory_changed(self):
        """ This will update the model """

        try:
            self.setup['directory'] = self.lineEdit_6.text()
            files = [f for f in sorted(os.listdir(self.setup['directory'])) if self.setup['file_suffix'] in f]
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

        self.on_setup_changed()

        try:
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
                psr = self.setup['peak_search_radius']
                upp = self.setup['units_per_pixel']

                sl = np.s_[:, int(pos - wi/2) : int(pos + wi/2)]

                img = sobel_convolution(plt.imread(file_full)[sl])

                # plt.figure()
                # plt.imshow(img, aspect = "auto")
                # plt.show()

                if self.setup['transpose_image']:
                    img = img.T

                profile =  img.sum(axis = 0)



                # Determine jet width
                c = np.ma.masked_array(profile)
                x = np.arange(len(profile))
                mask = np.array([False] * len(c))
                mask[:ec] = True
                mask[-ec:] = True
                c.mask = mask
                y0, y1 = np.min(c), np.max(c)
                i0 = np.argmax(c)
                print(i0)
                pos0 = calculateCOM(x, c - np.nanmin(c), window = [i0 - psr, i0 + psr + 1])
                mask[i0-psr:i0+psr+1] = True
                c.mask = mask

                i1 = np.argmax(c)
                pos1 = calculateCOM(x, c - np.nanmin(c), window = [i1 - psr, i1 + psr + 1])

                width = float(np.abs(pos1 - pos0)) * upp

                item_d.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled  | QtCore.Qt.ItemIsEditable)
                item_d.setData(width , role = QtCore.Qt.DisplayRole)
                item_d.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

                app.processEvents()

                if self.setup['plot']:
                    plt.figure()
                    plt.plot(np.arange(len(profile)) * self.setup['units_per_pixel'], profile, label = file)
                    # plt.plot([pos0 * upp, pos0 * upp], [y0, y1],  "r")
                    # plt.plot([pos1 * upp , pos1 * upp], [y0, y1],  "r")
                    # plt.plot([i0 * upp, i0 * upp], [y0, y1],  "b--")
                    # plt.plot([i1 * upp, i1 * upp], [y0, y1],  "b--")
                    plt.ylim([y0 - 0.1 * np.ptp([y0,y1]), y1 + 0.1 * np.ptp([y0,y1])])
                    plt.xlim([pos0 * upp - 2 *width, pos1*upp + 2*width ])


            self.progressBar.setValue(100.)

            if self.setup['plot']:
                plt.show()
                plt.legend()

        except Exception as e:
            print(e)





    def on_setup_changed(self):
        try:
            self.setup['slice_position'] = int(self.slicePositionLineEdit.text())
            self.setup['slice_width'] = int(self.sliceWidthLineEdit.text())
            self.setup['image_width'] = int(self.imageWidthPxLineEdit.text())
            self.setup['image_height'] = int(self.imageHeightPxLineEdit.text())
            self.setup['units_per_pixel'] = float(self.unitsPerPixelLineEdit.text())
            self.setup['transpose_image'] = self.transposeImageCheckBox.isChecked() == True
            self.setup['edge_crop'] = int(self.edgeCropPxLineEdit.text())
            self.setup['peak_search_radius'] = int(self.peakSearchRadiusLineEdit.text())
            self.setup['file_suffix'] = self.fileSuffixLineEdit.text()
            self.setup['plot'] = self.showPlotCheckBox.isChecked() == True


        except Exception as e:
            print("Invalid setup state: {}".format(e))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ADT_MainWindow()
    window.show()
    sys.exit(app.exec_())
