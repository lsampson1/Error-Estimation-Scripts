########################################################################################################################
# Purpose: Run.py, calls and runs all necessary python scripts for producing error estimation component for
#
# Prerequisites:
#       For preprocessing.
#       - BaseDir available.
#       - YYYYMMDDT0000Z_xxx_qc_BiasCorrfb_oo_qc_fdbk.nc, for the set date and variable under the BaseDir.
#       - rossy_radii.nc, from NEMOVar suite. (Original is in ocean\OPERATIONAL_SUITE_V5.3\...)
#       - Error variance, length-scales and obs netCDF files. Examples for formatting output on model grid.
#         Bmod_sdv_mxl... and Bmod_sdv_wgt... (This is to format .nc files)
#       - Unzipped assimilated model runs. e.g. diaopfoam files (This is for depth gradient). UZ.sh script will unzip.
#       - BaseDir should contain ocean\OPERATIONAL_SUITE_V5.3\...
#       For running H-L/IPA.
#       - Seasonal coarse grid. BaseDir\Preprocessed\Season
#       - Innovations for entire season. BaseDir\Preprocessed\Innovations_Typ
#       - Seasonal Rossby radius. BaseDir\Preprocessed\Season
#
# Output:
#       For preprocessing
#       - Stored innovations from Start to End-1 day.
#       - Coarse masked grid, Rossby radius and TGradient for Seasons.
#       For H-L
#       - Background error variance, HL_Var.npy
#       - Observation error variance, HL_Obs.npy
#       - Background error length-scale ratio (wgt1), HL_Lsr.npy
#       - Script diagnostics, time run, observations used, etc.
#       - .jpg for Lsr, Sdv and Obs, with each %N, and each season. Both model and coarse grid.
#       - .nc for Lsr, Sdv, and Obs with 100% observations for each season. Only model grid.
#       For IPA
#       - Background error variance, IPA_Var.npy
#       - Observation error variance, IPA_Obs.npy
#       - Background error length-scale ratio (wgt1), IPA_Lsr.npy
#       - Script diagnostics, time run, observations used, etc.
########################################################################################################################
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog
import os
from Windows.mainwindow2 import Ui_MainWindow
from Windows.directoryerror import Ui_DirectoryError


class MainWindow(QDialog):
    def __init__(self, q):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.browseslot)
        self.ui.pushButton_2.clicked.connect(self.browseslot2)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.comboBox_2.currentIndexChanged['int'].connect(self.update)
        self.ui.comboBox.currentIndexChanged['int'].connect(self.update)
        self.ui.comboBox_3.currentIndexChanged['int'].connect(self.update)

        self.ui.lineEdit.setText(q[0])
        self.ui.lineEdit_2.setText(q[1])
        self.ui.comboBox_2.setCurrentIndex(q[2])
        self.ui.comboBox.setCurrentIndex(q[3])
        self.ui.comboBox_3.setCurrentText(q[4])

    def return_strings(self):
        return [self.ui.lineEdit.text(), self.ui.lineEdit_2.text(), self.ui.comboBox_2.currentIndex(),
                self.ui.comboBox.currentIndex(), self.ui.comboBox_3.currentText()]

    def browseslot(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        my_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select a folder",
            os.path.expanduser(os.path.dirname(self.ui.lineEdit.placeholderText())),
            options=options
        )
        if my_dir:
            self.ui.lineEdit.setPlaceholderText(my_dir)
            self.ui.lineEdit.setText(my_dir)
            self.ui.lineEdit.setCursorPosition(0)
            self.ui.lineEdit.setReadOnly(False)

    def browseslot2(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        my_dir = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select a folder",
            os.path.expanduser(os.path.dirname(self.ui.lineEdit_2.placeholderText())),
            options=options
        )
        if my_dir:
            self.ui.lineEdit_2.setPlaceholderText(my_dir)
            self.ui.lineEdit_2.setText(my_dir)
            self.ui.lineEdit_2.setCursorPosition(0)
            self.ui.lineEdit_2.setReadOnly(False)

    @staticmethod
    def get_data(q):
        dialog = MainWindow(q)
        a = dialog.exec_()
        if a == 1:
            return dialog.return_strings()
        if a == 0:
            print(40 * '==')
            print(14 * '~~' + ' Run cancelled by user  ' + 14 * '~~')
            print(40 * '==')
            exit(0)


class DirectoryError(QDialog):
    def __init__(self, s0):
        super(DirectoryError, self).__init__()
        self.ui = Ui_DirectoryError()
        self.ui.setupUi(self)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.label.setText(s0)


def main():
    app = QApplication([])
    q = ['Select Base Directory', 'Select Prerequisite Directory', 0, 0, 'SST']
    window = MainWindow(q)

    n = 0
    while n == 0:
        q = window.get_data(q)
        if os.path.isdir(q[0]) is False:
            print('Please choose/create a base directory')
            error = DirectoryError('Base directory unavailable. Please select another.')
            b = error.exec()
            if b == 0:
                print(60*'==')
                print(24*'~~' + ' Run cancelled by user  ' + 24*'~~')
                print(60*'==')
                exit(0)
        elif os.path.isdir(q[1]) is False:
            print('Please choose/create a prerequisite directory')
            error = DirectoryError('Prerequisite directory unavailable. Please select another.')
            b = error.exec()
            if b == 0:
                print(60*'==')
                print(24*'~~' + ' Run cancelled by user  ' + 24*'~~')
                print(60*'==')
                exit(0)
        else:
            n = 1

    if q[2] == 0:
        from Preprocess import preprocessmain
        preprocessmain(q[0], q[1], q[4].lower())

    if q[3] == 0:
        from HL import hlmain
        hlmain(q[0], q[4].lower())
        from Outputs import outputsmain
        outputsmain(q[0], q[4].lower(), ['HL'])
    elif q[3] == 1:
        from IPA import ipamain
        ipamain(q[0], q[4].lower())
        from Outputs import outputsmain
        outputsmain(q[0], q[4].lower(), ['IPA'])
    else:
        from HL import hlmain
        hlmain(q[0], q[4].lower())
        from IPA import ipamain
        ipamain(q[0], q[4].lower())
        from Outputs import outputsmain
        outputsmain(q[0], q[1], q[4].lower(), ['IPA', 'HL'])


if __name__ == "__main__":
    main()
