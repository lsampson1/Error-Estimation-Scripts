# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(429, 252)
        MainWindow.setSizeGripEnabled(False)
        MainWindow.setModal(False)
        self.gridLayout_3 = QtWidgets.QGridLayout(MainWindow)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_2 = QtWidgets.QFrame(MainWindow)
        self.frame_2.setAutoFillBackground(True)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_2.setEnabled(True)
        self.lineEdit_2.setText("")
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_6.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.frame_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_6.addWidget(self.label_13, 0, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_6.addWidget(self.pushButton, 0, 2, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit.setEnabled(True)
        self.lineEdit.setText("")
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_6.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_6.addWidget(self.label_14, 1, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_2.setMouseTracking(True)
        self.pushButton_2.setWhatsThis("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_6.addWidget(self.pushButton_2, 1, 2, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 0, 0, 1, 1)
        self.frame = QtWidgets.QFrame(MainWindow)
        self.frame.setAutoFillBackground(True)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(300, 0))
        self.label.setMaximumSize(QtCore.QSize(16777215, 16777197))
        self.label.setAutoFillBackground(True)
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label.setScaledContents(True)
        self.label.setWordWrap(True)
        self.label.setIndent(0)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.frame)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_2, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMinimumSize(QtCore.QSize(300, 0))
        self.label_2.setAutoFillBackground(True)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.frame)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_2.addWidget(self.comboBox, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 2, 0, 1, 1)
        self.comboBox_3 = QtWidgets.QComboBox(self.frame)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_3, 2, 1, 1, 1)
        self.gridLayout_3.addWidget(self.frame, 1, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(MainWindow)
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame_3)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_3, 2, 0, 1, 1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Define Parameters"))
        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", "~/test"))
        self.label_13.setText(_translate("MainWindow", "Base Directory"))
        self.pushButton.setText(_translate("MainWindow", "Browse"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "~/test"))
        self.label_14.setText(_translate("MainWindow", "Prerequisite Directory"))
        self.pushButton_2.setText(_translate("MainWindow", "Browse"))
        self.label.setText(_translate("MainWindow", "Do you require preprocessing:"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Y"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "N"))
        self.label_2.setText(_translate("MainWindow", "Error estimation method:"))
        self.comboBox.setItemText(0, _translate("MainWindow", "H-L"))
        self.comboBox.setItemText(1, _translate("MainWindow", "IPA"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Both"))
        self.label_5.setText(_translate("MainWindow", "Variable:"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "SST"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "SLA"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

