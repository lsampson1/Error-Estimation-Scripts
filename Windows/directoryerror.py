# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'directoryerror.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DirectoryError(object):
    def setupUi(self, DirectoryError):
        DirectoryError.setObjectName("DirectoryError")
        DirectoryError.resize(435, 86)
        DirectoryError.setFocusPolicy(QtCore.Qt.NoFocus)
        self.frame = QtWidgets.QFrame(DirectoryError)
        self.frame.setGeometry(QtCore.QRect(10, 9, 421, 71))
        self.frame.setAutoFillBackground(True)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(1)
        self.frame.setObjectName("frame")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame)
        self.buttonBox.setGeometry(QtCore.QRect(330, 10, 81, 51))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(10, 5, 311, 61))
        self.label.setObjectName("label")

        self.retranslateUi(DirectoryError)
        self.buttonBox.accepted.connect(DirectoryError.accept)
        self.buttonBox.rejected.connect(DirectoryError.reject)
        QtCore.QMetaObject.connectSlotsByName(DirectoryError)

    def retranslateUi(self, DirectoryError):
        _translate = QtCore.QCoreApplication.translate
        DirectoryError.setWindowTitle(_translate("DirectoryError", "Directory Error"))
        self.label.setText(_translate("DirectoryError", "Base directory unavailable. Please select another."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DirectoryError = QtWidgets.QDialog()
    ui = Ui_DirectoryError()
    ui.setupUi(DirectoryError)
    DirectoryError.show()
    sys.exit(app.exec_())

