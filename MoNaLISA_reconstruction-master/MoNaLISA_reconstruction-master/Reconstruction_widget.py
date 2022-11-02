# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:56:03 2018

@author: andreas.boden
"""

import os
import sys
import time
import psutil

import numpy as np
import matplotlib.pyplot as plt

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5' #force Qt to use PyQt5
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
if not os.environ['PY_UTILS_PATH'] in sys.path:
    sys.path.append(os.environ['PY_UTILS_PATH'])

import DataIO_tools
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph as pg

import ctypes
import h5py
import tifffile as tiff
import copy

import Pattern_finder

#Add path to dependency DLLs.
os.environ['PATH'] = os.environ['PATH']  + ';' + os.path.join(os.getcwd(), 'dlls')
# Recipy from:
# https://code.activestate.com/recipes/460509-get-the-actual-and-usable-sizes-of-all-the-monitor/
user = ctypes.windll.user32
pg.setConfigOption('imageAxisOrder', 'row-major')


class RECT(ctypes.Structure):
    _fields_ = [
            ('left', ctypes.c_long),
            ('top', ctypes.c_long),
            ('right', ctypes.c_long),
            ('bottom', ctypes.c_long)
            ]

    def dump(self):
        return map(int, (self.left, self.top, self.right, self.bottom))


def n_monitors():
    retval = []
    CBFUNC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                                ctypes.POINTER(RECT), ctypes.c_double)

    def cb(hMonitor, hdcMonitor, lprcMonitor, dwData):
        r = lprcMonitor.contents
        data = [hMonitor]
        data.append(r.dump())
        retval.append(data)
        return 1
    cbfunc = CBFUNC(cb)
    temp = user.EnumDisplayMonitors(0, 0, cbfunc, 0)

    return len(retval)


class ReconParTree(ParameterTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parameter tree for the reconstruction
        params = [
            {'name': 'Pixel size', 'type': 'float', 'value': 65, 'suffix': 'nm'},
            {'name': 'CPU/GPU', 'type': 'list', 'values': ['GPU', 'CPU']},
            {'name': 'Pattern', 'type': 'group', 'children': [
                {'name': 'Row-offset', 'type': 'float', 'value': 9.89, 'limits': (0, 9999)},
                {'name': 'Col-offset', 'type': 'float', 'value': 10.4, 'limits': (0, 9999)},
                {'name': 'Row-period', 'type': 'float', 'value': 11.05, 'limits': (0, 9999)},
                {'name': 'Col-period', 'type': 'float', 'value': 11.05, 'limits': (0, 9999)},
                {'name': 'Find pattern', 'type': 'action'}]},
            {'name': 'Reconstruction options', 'type': 'group', 'children': [
                {'name': 'PSF FWHM', 'type': 'float', 'value': 220, 'limits': (0,9999), 'suffix': 'nm'},
                {'name': 'BG modelling', 'type': 'list', 'values': ['Constant', 'Gaussian', 'No background'], 'children': [
                    {'name': 'BG Gaussian size', 'type': 'float', 'value': 500, 'suffix': 'nm'}]}]},
            {'name': 'Scanning parameters', 'type': 'action'},
            {'name': 'Show pattern', 'type': 'bool'}]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True


class ReconWid(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('MoNaLISA Reconstruction')
        self.setWindowIcon(QtGui.QIcon(r'/Graphics/ML_logo.ico'))
#        self.showFullScreen()
        """self parameters"""
        self.r_l_text = 'Right/Left'
        self.u_d_text = 'Up/Down'
        self.b_f_text = 'Back/Forth'
        self.timepoints_text = 'Timepoints'
        self.p_text = 'pos'
        self.n_text = 'neg'

        self.current_data = None

        # Actions in menubar
        menubar = self.menuBar()
        File = menubar.addMenu('&File')

        saveReconAction = QtGui.QAction('Save reconstruction', self)
        saveReconAction.setShortcut('Ctrl+S')
        saveReconAction.triggered.connect(lambda: self.save_current('reconstruction'))
        File.addAction(saveReconAction)
        saveCoeffsAction = QtGui.QAction('Save coefficients', self)
        saveCoeffsAction.setShortcut('Ctrl+A')
        saveCoeffsAction.triggered.connect(lambda: self.save_current('coefficients'))
        File.addAction(saveCoeffsAction)

        setDataFolder = QtGui.QAction('Set data folder', self)
        setDataFolder.triggered.connect(self.SetDataFolder)
        File.addAction(setDataFolder)

        setSaveFolder = QtGui.QAction('Set save folder', self)
        setSaveFolder.triggered.connect(self.SetSaveFolder)
        File.addAction(setSaveFolder)

        self.extractor = SignalExtractor('dlls\GPU_acc_recon.dll')
        self.pat_finder = Pattern_finder.pattern_finder()
        self.data_frame = Data_Frame(self)
        self.multi_data_frame = MultiDataFrame()
        self.multi_data_frame.currentDataChangedSig.connect(self.changeAndShowCurrentData)
        btn_frame = BtnFrame(self)
        btn_frame.recon_curr_sig.connect(self.reconstruct_current)
        btn_frame.recon_multi_sig.connect(self.reconstruct_multi)
        btn_frame.q_load_data_sig.connect(self.quick_load_data)
        btn_frame.update_sig.connect(self.update)

        self.recon_frame = Recon_Frame()

        self.partree = ReconParTree()
        self.scanningParDict= {'dimensions': [self.r_l_text, self.u_d_text, self.b_f_text, self.timepoints_text],
                               'directions': [self.p_text, self.p_text, self.p_text],
                               'steps': ['35', '35', '1', '1'],
                               'step_sizes': ['35', '35', '35', '1'],
                               'unidirectional': True}
        self.scanningParWindow = ScanningParWindow(self,
                                                   self.scanningParDict,
                                                   self.r_l_text,
                                                   self.u_d_text,
                                                   self.b_f_text,
                                                   self.timepoints_text,
                                                   self.p_text,
                                                   self.n_text)

        parameterFrame = QtGui.QFrame()
        parameterGrid = QtGui.QGridLayout()
        parameterFrame.setLayout(parameterGrid)
        parameterGrid.addWidget(self.partree, 0, 0)

        DataDock = DockArea()

        MultiDataDock = Dock('Multidata management')
        MultiDataDock.addWidget(self.multi_data_frame)
        DataDock.addDock(MultiDataDock)

        CurrentDataDock = Dock('Current data')
        CurrentDataDock.addWidget(self.data_frame)
        DataDock.addDock(CurrentDataDock, 'above', MultiDataDock)

        layout = QtGui.QGridLayout()
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        self.cwidget.setLayout(layout)

#        if n_monitors() == 1:
        layout.setColumnMinimumWidth(0, 500)
        layout.setColumnMinimumWidth(1, 1500)
        layout.setRowMinimumHeight(0, 500)
        layout.setRowMinimumHeight(1, 500)

        layout.addWidget(parameterFrame, 0, 0)
        layout.addWidget(btn_frame, 1, 0)
        layout.addWidget(DataDock, 2, 0)
        layout.addWidget(self.recon_frame, 0, 1, 3, 1)

        layout.setRowMinimumHeight(0, 400)
        layout.setRowMinimumHeight(1, 50)
        layout.setRowMinimumHeight(2, 900)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 1)

        pg.setConfigOption('imageAxisOrder', 'row-major')

        self.show_pat_bool = self.partree.p.param('Show pattern')
        self.show_pat_bool.sigStateChanged.connect(self.toggle_pattern)
        self.find_pat_btn = self.partree.p.param('Pattern').param('Find pattern')
        self.find_pat_btn.sigStateChanged.connect(self.find_pattern)
        self.scanParWin_btn = self.partree.p.param('Scanning parameters')
        self.scanParWin_btn.sigStateChanged.connect(self.update_scanning_pars)

        self.update_pattern()

        self.partree.p.param('Pattern').sigTreeStateChanged.connect(self.update_pattern)


    def test(self):
        print('Test fcn run')

    def update_scanning_pars(self):
        self.scanningParWindow.show()

    def find_pattern(self):
        print('Find pattern clicked')
        if not self.data_frame.mean_data == []:
            print('Finding pattern')
            im = self.data_frame.mean_data
            pattern = self.pat_finder.find_pattern(im)
            print('Pattern found as: ', self.pattern)
            pattern_pars = self.partree.p.param('Pattern')
            pattern_pars.param('Row-offset').setValue(pattern[0])
            pattern_pars.param('Col-offset').setValue(pattern[1])
            pattern_pars.param('Row-period').setValue(pattern[2])
            pattern_pars.param('Col-period').setValue(pattern[3])
            self.update_pattern()

    def SetDataFolder(self):
        self.data_folder = QtGui.QFileDialog.getExistingDirectory()
        self.multi_data_frame.data_folder= self.data_folder

    def SetSaveFolder(self):
        self.save_folder = QtGui.QFileDialog.getExistingDirectory()

    def toggle_pattern(self):
        print('Toggling pattern')
        if self.show_pat_bool.value():
            self.data_frame.pattern = self.pattern
            self.data_frame.show_pat = True
        else:
            self.data_frame.show_pat = False

    def update_pattern(self):
        print('Updating pattern')
        pattern_pars = self.partree.p.param('Pattern')
        self.pattern = [np.mod(pattern_pars.param('Row-offset').value(), pattern_pars.param('Row-period').value()),
                        np.mod(pattern_pars.param('Col-offset').value(), pattern_pars.param('Col-period').value()),
                        pattern_pars.param('Row-period').value(),
                        pattern_pars.param('Col-period').value()]

        if self.data_frame.show_pat:
            self.data_frame.pattern = self.pattern
            self.data_frame.make_pattern_grid()

    def update(self):

        self.recon_frame.UpdateScanPars(self.scanningParDict)


    def quick_load_data(self):

        dlg = QtGui.QFileDialog()
        if hasattr(self, 'data_folder'):
            datapath = dlg.getOpenFileName(directory=self.data_folder)[0]
        else:
            datapath = dlg.getOpenFileName()[0]

        if datapath:
            print('Loading data at:', datapath)


            name = os.path.split(datapath)[1]
            if not self.current_data is None:
                self.current_data.checkAndUnloadData()
            self.current_data = DataObj(name, datapath)
            self.current_data.checkAndLoadData()
            if self.current_data.data_loaded == True:
                self.data_frame.setData(self.current_data)
                self.multi_data_frame.allWhite()
                print('Data loaded')
            else:
                pass


    def changeAndShowCurrentData(self):
        self.currDataChanged()
        self.showCurrentData()

    def currDataChanged(self):

        newCurrDataObj = self.multi_data_frame.data_list.currentItem().data(1)
        newCurrDataObj.checkAndLoadData()

        self.current_data = newCurrDataObj

    def showCurrentData(self):
        self.data_frame.setData(self.current_data)

    def extract_data(self):

        recon_pars = self.partree.p.param('Reconstruction options')
        fwhm_nm = recon_pars.param('PSF FWHM').value()
        if recon_pars.param('BG modelling').value() == 'Constant':
            fwhm_nm = np.append(fwhm_nm, 9999) #Code for constant bg
        elif recon_pars.param('BG modelling').value() == 'No background':
            fwhm_nm = np.append(fwhm_nm, 0) #Code for zero bg
        else:
            print('In Gaussian version')
            fwhm_nm = np.append(fwhm_nm, recon_pars.param('BG modelling').param('BG Gaussian size').value())
            print('Appended to sigmas')

        sigmas = np.divide(
            fwhm_nm, 2.355*self.partree.p.param('Pixel size').value())


        if self.partree.p.param('CPU/GPU').value() == 'CPU':
            coeffs = self.extractor.extract_signal(self.current_data.data, sigmas, self.pattern, 'cpu')
        elif self.partree.p.param('CPU/GPU').value() == 'GPU':
            coeffs = self.extractor.extract_signal(self.current_data.data, sigmas, self.pattern, 'gpu')

        return coeffs

    def reconstruct_current(self):
        if self.current_data is None:
            pass
        elif np.prod(np.array(self.scanningParDict['steps'], dtype=np.int)) < self.current_data.frames:
            print('Too many frames in data')
        else:
            coeffs = self.extract_data()

            reconObj = ReconObj(self.current_data.name,
                                self.scanningParDict,
                                self.r_l_text,
                                self.u_d_text,
                                self.b_f_text,
                                self.timepoints_text,
                                self.p_text,
                                self.n_text)
            reconObj.addCoeffsTP(coeffs)
            reconObj.update_images()

            self.recon_frame.AddNewData(reconObj)

    def reconstruct_multi(self):
        data_list = self.multi_data_frame.data_list
        
        for i in np.arange(data_list.count()):
            self.current_data = data_list.item(i).data(1)
            pre_loaded = self.current_data.data_loaded
            self.current_data.checkAndLoadData()
            coeffs = self.extract_data()
            reconObj = ReconObj(self.current_data.name,
                    self.scanningParDict,
                    self.r_l_text,
                    self.u_d_text,
                    self.b_f_text,
                    self.timepoints_text,
                    self.p_text,
                    self.n_text)
            if not pre_loaded:
                data_list.item(i).data(1).checkAndUnloadData()
            reconObj.addCoeffsTP(coeffs)
            reconObj.update_images()
            self.recon_frame.AddNewData(reconObj)

    def save_current(self, data_type=None):
        """Saves the reconstructed image from self.reconstructor to specified
        destination"""
        if data_type:
            dlg = QtGui.QFileDialog()
            if hasattr(self, 'save_folder'):
                savename = dlg.getSaveFileName(self, 'Save File', filter='*.tiff', directory=self.save_folder)[0]
            else:
                savename = dlg.getSaveFileName(self, 'Save File', filter='*.tiff')[0]
            print(savename)
            if savename:
                if data_type == 'reconstruction':
                    reconstruction_obj = self.recon_frame.recon_list.currentItem().data(1)
                    vxsizec = int(reconstruction_obj.scanningParDict['step_sizes'][self.scanningParDict['dimensions'].index(self.r_l_text)])
                    vxsizer = int(reconstruction_obj.scanningParDict['step_sizes'][self.scanningParDict['dimensions'].index(self.u_d_text)])
                    vxsizez = int(reconstruction_obj.scanningParDict['step_sizes'][self.scanningParDict['dimensions'].index(self.b_f_text)])
                    dt = int(reconstruction_obj.scanningParDict['step_sizes'][self.scanningParDict['dimensions'].index(self.timepoints_text)])

                    print('Trying to save to: ', savename,'Vx size:', vxsizec, vxsizer, vxsizez)
                    # Reconstructed image
                    reconstr_data = copy.deepcopy(reconstruction_obj.getReconstruction())
                    reconstr_data = reconstr_data[:,0,:,:,:,:]
                    reconstr_data.shape =  reconstr_data.shape[0], reconstr_data.shape[1], reconstr_data.shape[2], reconstr_data.shape[3], reconstr_data.shape[4], 1
                    reconstr_data = np.swapaxes(reconstr_data, 1, 2)
                    tiff.imwrite(savename, reconstr_data,
                                imagej=True, resolution=(1/vxsizec, 1/vxsizer),
                                metadata={'spacing': vxsizez, 'unit': 'nm', 'axes': 'TZCYX'})
                elif data_type == 'coefficients':
                    coeffs = copy.deepcopy(self.recon_frame.getSelectedCoeffs())
                    print('Shape of coeffs = ', coeffs.shape)
                    try:
                        coeffs = np.swapaxes(coeffs, 1, 2)
                        tiff.imwrite(savename, coeffs,
                                     imagej=True, resolution=(1, 1),
                                     metadata={'spacing': 1, 'unit': 'px', 'axes': 'TZCYX'})
                    except:
                        pass
                else:
                    print('Data type in save_current not recognized')
            else:
                print('No saving path given')
        else:
            print('No data type given in save current')

class BtnFrame(QtWidgets.QFrame):
    recon_curr_sig = QtCore.pyqtSignal()
    recon_multi_sig = QtCore.pyqtSignal()
    q_load_data_sig = QtCore.pyqtSignal()
    update_sig = QtCore.pyqtSignal()
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        recon_curr_btn = QtWidgets.QPushButton('Reconstruct current')
        recon_curr_btn.clicked.connect(self.recon_curr_sig.emit)
        recon_multi_btn = QtWidgets.QPushButton('Reconstruct multidata')
        recon_multi_btn.clicked.connect(self.recon_multi_sig.emit)
        q_load_data_btn = QtWidgets.QPushButton('Quick load data')
        q_load_data_btn.clicked.connect(self.q_load_data_sig.emit)
        update_btn = QtWidgets.QPushButton('Update reconstruction')
        update_btn.clicked.connect(self.update_sig.emit)



        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(q_load_data_btn, 0, 0, 1, 2)
        layout.addWidget(recon_curr_btn, 1, 0)
        layout.addWidget(recon_multi_btn, 1, 1)
        layout.addWidget(update_btn, 2, 0, 1, 2)

class Data_Frame(QtGui.QFrame):
    """Frame for showing and examining the raw data"""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataObj = None
        self.mean_data = []
        self.pattern = []
        self.pattern_grid = []
        self.data_edit = DataEdit(parent)

        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.img_vb = imageWidget.addViewBox(row=0, col=0)
        self.img_vb.setMouseMode(pg.ViewBox.PanMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.img_vb.addItem(self.img)
        self.img_vb.setAspectLocked(True)
        self.img_hist = pg.HistogramLUTItem(image=self.img)
        imageWidget.addItem(self.img_hist, row=0, col=1)

        self.show_mean_btn = QtGui.QPushButton()
        self.show_mean_btn.setText('Show mean image')
        self.show_mean_btn.pressed.connect(self.show_mean)

        self.AdjustDataBtn = QtGui.QPushButton()
        self.AdjustDataBtn.setText('Adjust/compl. data')
        self.AdjustDataBtn.pressed.connect(self.adjustData)

        self.UnloadDataBtn = QtGui.QPushButton()
        self.UnloadDataBtn.setText('Unload data')
        self.UnloadDataBtn.pressed.connect(self.unloadData)

        frame_label = QtGui.QLabel('Frame # ')
        self.frame_nr = QtGui.QLineEdit('0')
        self.frame_nr.textChanged.connect(self.setImgSlice)
        self.frame_nr.setFixedWidth(45)

        data_name_label = QtWidgets.QLabel('File name:')
        self.data_name = QtWidgets.QLabel('')
        nr_frames_label = QtWidgets.QLabel('Nr of frames:')
        self.nr_frames = QtWidgets.QLabel('')

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setTickInterval(5)
        self.slider.setSingleStep(1)
        self.slider.valueChanged[int].connect(self.slider_moved)

        self.pattern_scatter = pg.ScatterPlotItem()
        self.pattern_scatter.setData(
            pos=[[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]],
            pen=pg.mkPen(color=(255, 0, 0), width=0.5,
                         style=QtCore.Qt.SolidLine, antialias=True),
            brush=pg.mkBrush(color=(255, 0, 0), antialias=True), size=1,
            pxMode=False)

        self.editWdw = DataEdit()

        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(data_name_label, 0, 0)
        layout.addWidget(self.data_name, 0, 1)
        layout.addWidget(nr_frames_label, 0, 2)
        layout.addWidget(self.nr_frames, 0, 3)
        layout.addWidget(self.show_mean_btn, 1, 0)
        layout.addWidget(self.slider, 1, 1)
        layout.addWidget(frame_label, 1, 2)
        layout.addWidget(self.frame_nr, 1, 3)
        layout.addWidget(self.AdjustDataBtn, 2, 0)
        layout.addWidget(self.UnloadDataBtn, 2, 1)
        layout.addWidget(imageWidget, 3, 0, 1, 4)

        self._show_pat = False
        self.pat_grid_made = False


    @property
    def show_pat(self):
        return self._show_pat

    @show_pat.setter
    def show_pat(self, b_value):
        if b_value:
            self._show_pat = True
            print('Showing pattern')
            if not self.pat_grid_made:
                self.make_pattern_grid()
            self.img_vb.addItem(self.pattern_scatter)
        else:
            print('Hiding pattern')
            self._show_pat = False
            self.img_vb.removeItem(self.pattern_scatter)

    def slider_moved(self):
        self.frame_nr.setText(str(self.slider.value()))
        self.setImgSlice()

    def setImgSlice(self):
        try:
            i = int(self.frame_nr.text())
        except TypeError:
            print('ERROR: Input must be an integer value')

        self.slider.setValue(i)
        self.img.setImage(self.dataObj.data[i], autoLevels=False)
        self.frame_nr.setText(str(i))

    def unloadData(self):
        self.mean_data = np.zeros([100, 100])
        self.dataObj = None
        self.show_mean()
        self.data_frames = 0
        self.nr_frames.setText('')
        self.data_name.setText('')
        self.slider.setMaximum(0)

    def adjustData(self):
        print('In adjust data')
        if self.dataObj is not None:
            self.editWdw.setData(self.dataObj)
            self.editWdw.show()
        else:
            print('No data to edit')

    def show_mean(self):
        self.img.setImage(self.mean_data)

    def setData(self, in_dataObj):
        self.dataObj = in_dataObj
        print('Data shape = ', self.dataObj.data.shape)
        self.mean_data = np.array(np.mean(self.dataObj.data, 0), dtype=np.float32)
        self.show_mean()
        self.data_frames = self.dataObj.frames
        self.nr_frames.setText(str(self.data_frames))
        self.data_name.setText(self.dataObj.name)
        self.slider.setMaximum(self.data_frames - 1)

    def make_pattern_grid(self):
        """ Pattern is now [Row-offset, Col-offset, Row-period, Col-period] where
        offset is calculated from the upper left corner (0, 0), while the
        scatter plot plots from lower left corner, so a flip has to be made
        in rows."""
        nr_cols = np.size(self.dataObj.data, 1)
        nr_rows = np.size(self.dataObj.data, 2)
        nr_points_col = int(1 + np.floor(((nr_cols - 1) - self.pattern[1]) / self.pattern[3]))
        nr_points_row = int(1 + np.floor(((nr_rows - 1) - self.pattern[0]) / self.pattern[2]))
        col_coords = np.linspace(self.pattern[1], self.pattern[1] + (nr_points_col - 1)*self.pattern[3], nr_points_col)
        row_coords = np.linspace(self.pattern[0], self.pattern[0] + (nr_points_row - 1)*self.pattern[2], nr_points_row)
        col_coords = np.repeat(col_coords, nr_points_row)
        row_coords = np.tile(row_coords, nr_points_col)
        self.pattern_grid = [col_coords, row_coords]
        self.pattern_scatter.setData(x=self.pattern_grid[0], y=self.pattern_grid[1])
        self.pat_grid_made = True
        print('Made new pattern grid')


class MultiDataFrame(QtGui.QFrame):
    """Signals"""
    currentDataChangedSig = QtCore.pyqtSignal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_list = QtGui.QListWidget()
        self.data_list.currentItemChanged.connect(self.UpdateInfo)
        self.data_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        data_loaded_label = QtGui.QLabel('Data loaded')
        data_loaded_label.setAlignment(QtCore.Qt.AlignTop)
        self.data_loaded_status = QtGui.QLabel()
        self.data_loaded_status.setAlignment(QtCore.Qt.AlignTop)

        setDataBtn = QtGui.QPushButton('Set ass current data')
        setDataBtn.clicked.connect(self.setCurrentData)
        addDataBtn = QtGui.QPushButton('Add data')
        addDataBtn.clicked.connect(self.addData)
        loadCurrDataBtn = QtGui.QPushButton('Load chosen data')
        loadCurrDataBtn.clicked.connect(self.loadCurrData)
        loadAllDataBtn = QtGui.QPushButton('Load all data')
        loadAllDataBtn.clicked.connect(self.loadAllData)

        delDataBtn = QtGui.QPushButton('Delete')
        delDataBtn.clicked.connect(self.delData)
        unloadDataBtn = QtGui.QPushButton('Unload')
        unloadDataBtn.clicked.connect(self.unloadData)
        delAllDataBtn = QtGui.QPushButton('Deleta all')
        delAllDataBtn.clicked.connect(self.delAllData)
        unloadAllDataBtn = QtGui.QPushButton('Unload all')
        unloadAllDataBtn.clicked.connect(self.unloadAllData)

        RAMusageLabel = QtWidgets.QLabel('Current RAM usage')

        self.memBar = QtGui.QProgressBar(self)
        self.memBar.setMaximum(100) #Percentage
        self.memBar.setValue(psutil.virtual_memory()[2])

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateMemBar)
        self.timer.start(1000)

        """Set layout"""
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(self.data_list, 0, 0, 4, 1)
        layout.addWidget(data_loaded_label, 0, 1)
        layout.addWidget(self.data_loaded_status, 0, 2)
        layout.addWidget(addDataBtn, 1, 1)
        layout.addWidget(loadCurrDataBtn, 2, 1)
        layout.addWidget(loadAllDataBtn, 3, 1)
        layout.addWidget(setDataBtn, 4, 1)
        layout.addWidget(delDataBtn, 1, 2)
        layout.addWidget(unloadDataBtn, 2, 2)
        layout.addWidget(delAllDataBtn, 3, 2)
        layout.addWidget(unloadAllDataBtn, 4, 2)
        layout.addWidget(RAMusageLabel, 4, 0)
        layout.addWidget(self.memBar, 5, 0)


    def addData(self):

        dlg = QtGui.QFileDialog()

        if hasattr(self, 'data_folder'):
            fileNames = dlg.getOpenFileNames(directory=self.data_folder)[0]
        else:
            fileNames = dlg.getOpenFileNames()[0]

        for i in np.arange(np.shape(fileNames)[0]):
            self.addDataObj(os.path.split(fileNames[i])[1], fileNames[i])

    def addDataObj(self, name, path):
        list_item = QtGui.QListWidgetItem('Data: ' + name)
        list_item.setData(1, DataObj(name, path))
        self.data_list.addItem(list_item)
        self.data_list.setCurrentItem(list_item)
        self.UpdateInfo()

    def loadCurrData(self):
        self.data_list.currentItem().data(1).checkAndLoadData()
        self.UpdateInfo()

    def loadAllData(self):
        for i in np.arange(self.data_list.count()):
            self.data_list.item(i).data(1).checkAndLoadData()
        self.UpdateInfo()

    def delData(self):
        nr_selected = np.shape(self.data_list.selectedIndexes())[0]
        while not nr_selected == 0:
            ind = self.data_list.selectedIndexes()[0]
            row = ind.row()
            removedItem = self.data_list.takeItem(row)
            nr_selected -= 1

    def unloadData(self):
        self.data_list.currentItem().data(1).checkAndUnloadData()
        self.UpdateInfo()

    def delAllData(self):
        for i in np.arange(self.data_list.count()):
            currRow = self.data_list.currentRow()
            removedItem = self.data_list.takeItem(currRow)

    def unloadAllData(self):
        for i in np.arange(self.data_list.count()):
            self.data_list.item(i).data(1).checkAndUnloadData()
        self.UpdateInfo()

    def setCurrentData(self):
        self.currentDataChangedSig.emit()

        self.allWhite()

        self.data_list.currentItem().setBackground(QtGui.QColor('green'))
        self.UpdateInfo()

    def UpdateInfo(self):
        if self.data_list.currentItem() is None:
            self.data_loaded_status.setText('')
        else:
            if self.data_list.currentItem().data(1).data_loaded:
                self.data_loaded_status.setText('Yes')
            else:
                self.data_loaded_status.setText('No')

    def allWhite(self):
        for i in np.arange(self.data_list.count()):
            self.data_list.item(i).setBackground(QtGui.QColor('white'))

    def updateMemBar(self):
        self.memBar.setValue(psutil.virtual_memory()[2])


class Recon_Frame(QtWidgets.QFrame):
    """ Frame for showing the reconstructed image"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#        self.data = np.array([])
        self.id = 0

        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.img_vb = imageWidget.addViewBox(row=0, col=0)
        self.img_vb.setMouseMode(pg.ViewBox.PanMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
#        self.img.setPxMode(True)
        self.img_vb.addItem(self.img)
        self.img_vb.setAspectLocked(True)
        self.img_hist = pg.HistogramLUTItem(image=self.img)
#        self.hist.vb.setLimits(yMin=0, yMax=2048)
        imageWidget.addItem(self.img_hist, row=0, col=1)

        """Slider and edit box for choosing slice"""
        slice_label = QtGui.QLabel('Slice # ')
        self.slice_nr = QtGui.QLabel('0')

        self.slice_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setTickInterval(5)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.valueChanged[int].connect(self.slice_slider_moved)

        base_label = QtGui.QLabel('Base # ')
        self.base_nr = QtGui.QLabel('0')

        self.base_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.base_slider.setMinimum(0)
        self.base_slider.setMaximum(0)
        self.base_slider.setTickInterval(5)
        self.base_slider.setSingleStep(1)
        self.base_slider.valueChanged[int].connect(self.base_slider_moved)

        self.time_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setTickInterval(5)
        self.time_slider.setSingleStep(1)
        self.time_slider.valueChanged[int].connect(self.time_slider_moved)

        time_label = QtGui.QLabel('Time point # ')
        self.time_nr = QtGui.QLabel('0')

        self.dataset_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.dataset_slider.setMinimum(0)
        self.dataset_slider.setMaximum(0)
        self.dataset_slider.setTickInterval(5)
        self.dataset_slider.setSingleStep(1)
        self.dataset_slider.valueChanged[int].connect(self.dataset_slider_moved)

        dataset_label = QtGui.QLabel('Dataset # ')
        self.dataset_nr = QtGui.QLabel('0')

        """Button group for choosing view"""
        self.choose_view_group = QtGui.QButtonGroup()
        self.choose_view_box = QtGui.QGroupBox('Choose view')
        self.view_layout = QtGui.QVBoxLayout()

        self.standard_view = QtGui.QRadioButton('Standard view')
        self.choose_view_group.addButton(self.standard_view, 3)
        self.view_layout.addWidget(self.standard_view)
        self.bottom_view = QtGui.QRadioButton('Bottom side view')
        self.choose_view_group.addButton(self.bottom_view, 4)
        self.view_layout.addWidget(self.bottom_view)
        self.left_view = QtGui.QRadioButton('Left side view')
        self.choose_view_group.addButton(self.left_view, 5)
        self.view_layout.addWidget(self.left_view)

        self.choose_view_box.setLayout(self.view_layout)

        self.choose_view_group.buttonClicked.connect(lambda: self.FullUpdate(levels=None))

        """List for storing sevral data sets"""
        self.recon_list = QtGui.QListWidget()
        self.recon_list.currentItemChanged.connect(self.list_item_changed)
        self.recon_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        removeReconBtn = QtWidgets.QPushButton('Remove current')
        removeReconBtn.clicked.connect(self.removeRecon)
        removeAllReconBtn = QtWidgets.QPushButton('Remove all')
        removeAllReconBtn.clicked.connect(self.removeAllRecon)

        self.currItem_ind = None

        """Set initial states"""
        self.standard_view.setChecked(True)

        """Set layout"""
        layout = QtGui.QGridLayout()

        self.setLayout(layout)

        layout.addWidget(imageWidget, 0, 0, 2, 1)
        layout.addWidget(self.choose_view_box, 0, 1, 1, 2)
        layout.addWidget(self.recon_list, 0, 3, 2, 1)
        layout.addWidget(self.slice_slider, 2, 0)
        layout.addWidget(slice_label, 2, 1)
        layout.addWidget(self.slice_nr, 2, 2)
        layout.addWidget(self.base_slider, 3, 0)
        layout.addWidget(base_label, 3, 1)
        layout.addWidget(self.base_nr, 3, 2)
        layout.addWidget(self.time_slider, 4, 0)
        layout.addWidget(time_label, 4, 1)
        layout.addWidget(self.time_nr, 4, 2)
        layout.addWidget(self.dataset_slider, 5, 0)
        layout.addWidget(dataset_label, 5, 1)
        layout.addWidget(self.dataset_nr, 5, 2)
        layout.addWidget(removeReconBtn, 2, 3)
        layout.addWidget(removeAllReconBtn, 3, 3)

        layout.setRowMinimumHeight(1, 800)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 100)
        layout.setColumnStretch(2, 5)

    def list_item_changed(self):

        if not self.currItem_ind is None:
            curr_hist_levels = self.img_hist.getLevels()
            prev_item = self.recon_list.item(self.currItem_ind).data(1)
            prev_item.setDispLevels(curr_hist_levels)
            retrieved_levels = self.recon_list.currentItem().data(1).getDispLevels()
            self.FullUpdate(levels=retrieved_levels)
            if not retrieved_levels is None:
                self.img_hist.setLevels(retrieved_levels[0], retrieved_levels[1])
            self.currItem_ind = self.recon_list.indexFromItem(self.recon_list.currentItem()).row()
        else:
            self.FullUpdate(levels=self.recon_list.currentItem().data(1).getDispLevels())
            self.currItem_ind = self.recon_list.indexFromItem(self.recon_list.currentItem()).row()

    def FullUpdate(self, levels):
        if self.recon_list.currentItem() is None:
            self.slice_slider.setValue(0)
            self.slice_slider.setMaximum(0)
            self.base_slider.setValue(0)
            self.base_slider.setMaximum(0)
            self.time_slider.setValue(0)
            self.time_slider.setMaximum(0)
            self.black_im()
        else:
            self.slice_slider.setValue(0)
            self.slice_slider.setMaximum(np.shape(self.recon_list.currentItem().data(1).reconstructed)[self.choose_view_group.checkedId()] - 1)
            self.dataset_slider.setValue(0)
            self.dataset_slider.setMaximum(np.shape(self.recon_list.currentItem().data(1).reconstructed)[0] - 1)
            self.base_slider.setValue(0)
            self.base_slider.setMaximum(np.shape(self.recon_list.currentItem().data(1).reconstructed)[1] - 1)
            self.time_slider.setValue(0)
            self.time_slider.setMaximum(np.shape(self.recon_list.currentItem().data(1).reconstructed)[2] - 1)
            self.setImgSlice(levels=levels)

    def slice_slider_moved(self):
        self.slice_nr.setText(str(self.slice_slider.value()))
        self.setImgSlice()

    def base_slider_moved(self):
        self.base_nr.setText(str(self.base_slider.value()))
        self.setImgSlice()

    def time_slider_moved(self):
        self.time_nr.setText(str(self.time_slider.value()))
        self.setImgSlice()

    def dataset_slider_moved(self):
        self.dataset_nr.setText(str(self.time_slider.value()))
        self.setImgSlice()

    def setImgSlice(self, autoLevels=False, levels=None):
        s = self.slice_slider.value()
        base = self.base_slider.value()
        t = self.time_slider.value()
        ds = self.dataset_slider.value()

        data = self.recon_list.currentItem().data(1).reconstructed
        if self.choose_view_group.checkedId() == 3:
            im = data[ds, base, t, s, ::, ::]
        elif self.choose_view_group.checkedId() == 4:
            im = data[ds, base, t, ::, s, ::]
        else:
            im = data[ds, t, base, ::, ::, s]

        if levels is None:
            self.img.setImage(im, autoLevels=autoLevels)
        else:
            self.img.setImage(im, levels=levels)

        self.slice_nr.setText(str(s))

    def getSelectedReconstruction(self):
        return self.recon_list.currentItem().data(1).getReconstruction()

    def getSelectedCoeffs(self):
        return self.recon_list.currentItem().data(1).getCoeffs()

    def AddNewData(self, recon_obj, name = None):
        if name is None:
            name = recon_obj.name
            ind = 0
            for i in np.arange(self.recon_list.count()):
                if name + '_' + str(ind) == self.recon_list.item(i).data(0):
                    ind += 1
            name = name + '_' + str(ind)

        list_item = QtGui.QListWidgetItem(name)
        list_item.setData(1, recon_obj)
        self.recon_list.addItem(list_item)
        self.recon_list.setCurrentItem(list_item)

    def removeRecon(self):
        nr_selected = np.shape(self.recon_list.selectedIndexes())[0]
        while not nr_selected == 0:
            ind = self.recon_list.selectedIndexes()[0]
            row = ind.row()
            removedItem = self.recon_list.takeItem(row)
            nr_selected -= 1

    def removeAllRecon(self):
        for i in np.arange(self.recon_list.count()):
            currRow = self.recon_list.currentRow()
            removedItem = self.recon_list.takeItem(currRow)

    def UpdateRecon(self):
        self.recon_list.currentItem().data(1).update_images()
        self.FullUpdate(levels=None)

    def UpdateScanPars(self, scanParDict):
        self.recon_list.currentItem().data(1).updateScanningPars(scanParDict)
        self.UpdateRecon()

    def black_im(self):
        self.img.setImage(np.zeros([100,100]))

class SignalExtractor(object):
    """ This class takes the raw data together with pre-set
    parameters and recontructs and stores the final images (for the different
    bases). Final images stored in
    - self.images

    """

    def __init__(self, dll_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images = np.array([])

        # This is needed by the DLL containing CUDA code.
#        ctypes.cdll.LoadLibrary(os.environ['CUDA_PATH_V9_0'] + '\\bin\\cudart64_90.dll')
        ctypes.cdll.LoadLibrary(os.path.join(os.getcwd(), 'dlls\cudart64_90.dll'))
        print(os.path.join(os.getcwd(), dll_path))
        self.ReconstructionDLL = ctypes.cdll.LoadLibrary(os.path.join(os.getcwd(), dll_path))

        self.data_shape_msg = QtGui.QMessageBox()
        self.data_shape_msg.setText(
            "Data does not have the shape of a square scan!")
        self.data_shape_msg.setInformativeText(
            "Do you want to append the data with tha last frame to enable "
            "reconstruction?")
        self.data_shape_msg.setStandardButtons(QtGui.QMessageBox.Yes |
                                               QtGui.QMessageBox.No)

    def make_3d_ptr_array(self, in_data):
        assert len(np.shape(in_data)) == 3, 'Trying to make 3D ctypes.POINTER array out of non-3D data'
        data = in_data
        slices = data.shape[0]

        pyth_ptr_array = []

        for j in np.arange(0, slices):
            ptr = data[j].ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            pyth_ptr_array.append(ptr)
        c_ptr_array = (ctypes.POINTER(ctypes.c_ubyte)*slices)(*pyth_ptr_array)
        return c_ptr_array

    def make_4d_ptr_array(self, in_data):
        assert len(np.shape(in_data)) == 4, 'Trying to make 4D ctypes.POINTER array out of non-4D data'
        data = in_data
        groups = data.shape[0]
        slices = data.shape[1]

        pyth_ptr_array = []

        for i in np.arange(0, groups):
            temp_p_ptr_array = []
            for j in np.arange(0, slices):
                ptr = data[i][j].ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                temp_p_ptr_array.append(ptr)
            temp_c_ptr_array = (ctypes.POINTER(ctypes.c_ubyte)*slices)(*temp_p_ptr_array)
            pyth_ptr_array.append(ctypes.cast(temp_c_ptr_array, ctypes.POINTER(ctypes.c_ubyte)))
        c_ptr_array = (ctypes.POINTER(ctypes.c_ubyte)*groups)(*pyth_ptr_array)

        return c_ptr_array

    def extract_signal(self, data, sigmas, pattern, dev):
        """Extracts the signal of the data according to given parameters.
        Output is a 4D matrix where first dimension is base and last three
        are frame and pixel coordinates."""

        print('Max in data = ', data.max())
        data_ptr_array = self.make_3d_ptr_array(data)
        p = ctypes.c_float*4
        # Minus one due to different (1 or 0) indexing in C/Matlab
        c_pattern = p(pattern[0], pattern[1], pattern[2], pattern[3])
        c_nr_bases = ctypes.c_int(np.size(sigmas))
        print('Sigmas = ', sigmas)
        sigmas = np.array(sigmas, dtype=np.float32)
        c_sigmas = np.ctypeslib.as_ctypes(sigmas)  # s(1, 10)
        c_grid_rows = ctypes.c_int(0)
        c_grid_cols = ctypes.c_int(0)
        c_im_rows = ctypes.c_int(data.shape[1])
        c_im_cols = ctypes.c_int(data.shape[2])
        c_im_slices = ctypes.c_int(data.shape[0])

        self.ReconstructionDLL.calc_coeff_grid_size(c_im_rows, c_im_cols, ctypes.byref(c_grid_rows), ctypes.byref(c_grid_cols), ctypes.byref(c_pattern))
        print('Coeff_grid calculated')
        res_coeffs = np.zeros(dtype=np.float32, shape=(c_nr_bases.value, c_im_slices.value, c_grid_rows.value, c_grid_cols.value))
        res_ptr = self.make_4d_ptr_array(res_coeffs)
        t = time.time()
        if dev == 'cpu':
            self.ReconstructionDLL.extract_signal_CPU(c_im_rows, c_im_cols, c_im_slices, ctypes.byref(c_pattern), c_nr_bases, ctypes.byref(c_sigmas), ctypes.byref(data_ptr_array), ctypes.byref(res_ptr))
        elif dev == 'gpu':
            self.ReconstructionDLL.extract_signal_GPU(c_im_rows, c_im_cols, c_im_slices, ctypes.byref(c_pattern), c_nr_bases, ctypes.byref(c_sigmas), ctypes.byref(data_ptr_array), ctypes.byref(res_ptr))
        elapsed = time.time() - t
        print('Signal extraction performed in', elapsed, 'seconds')
        return res_coeffs


class DataEditActions(QtWidgets.QFrame):
    setDarkFrame_sig = QtCore.pyqtSignal()
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        setDarkFrame_btn = QtWidgets.QPushButton('Set Dark/Offset frame')
        setDarkFrame_btn.clicked.connect(self.setDarkFrame_sig.emit)

        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(setDarkFrame_btn, 0, 0)



class DataEdit(QtGui.QMainWindow):
    """For future data editing window, for example to remove rearrange frames
    or devide into seperate datasets"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Data Edit/Complement')
        self.setWindowIcon(QtGui.QIcon(r'/Graphics/ML_logo.ico'))
        self.data = []

        # Data view Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.img_vb = imageWidget.addViewBox(row=0, col=0)
        self.img_vb.setMouseMode(pg.ViewBox.PanMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.img_vb.addItem(self.img)
        self.img_vb.setAspectLocked(True)
        self.img_hist = pg.HistogramLUTItem(image=self.img)
        imageWidget.addItem(self.img_hist, row=0, col=1)

        self.show_mean_btn = QtGui.QPushButton()
        self.show_mean_btn.setText('Show mean image')
        self.show_mean_btn.pressed.connect(self.show_mean)

        frame_label = QtGui.QLabel('Frame # ')
        self.frame_nr = QtGui.QLineEdit('0')
        self.frame_nr.textChanged.connect(self.setImgSlice)
        self.frame_nr.setFixedWidth(45)

        data_name_label = QtWidgets.QLabel('File name:')
        self.data_name = QtWidgets.QLabel('')
        nr_frames_label = QtWidgets.QLabel('Nr of frames:')
        self.nr_frames = QtWidgets.QLabel('')

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(np.shape(self.data)[0])
        self.slider.setTickInterval(5)
        self.slider.setSingleStep(1)
        self.slider.valueChanged[int].connect(self.slider_moved)

        self.actionBtns = DataEditActions(self)
        self.actionBtns.setDarkFrame_sig.connect(self.setDarkFrame)

        # Dark frame view widget
        DF_Widget = pg.GraphicsLayoutWidget()
        self.df_vb = DF_Widget.addViewBox(row=0, col=0)
        self.df_vb.setMouseMode(pg.ViewBox.PanMode)
        self.df = pg.ImageItem()
        self.df.translate(-0.5, -0.5)
        self.df_vb.addItem(self.df)
        self.df_vb.setAspectLocked(True)
        self.df_hist = pg.HistogramLUTItem(image=self.df)
        DF_Widget.addItem(self.df_hist, row=0, col=1)

        layout = QtGui.QGridLayout()
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        self.cwidget.setLayout(layout)

        layout.addWidget(data_name_label, 0, 0)
        layout.addWidget(self.data_name, 0, 1)
        layout.addWidget(nr_frames_label, 0, 2)
        layout.addWidget(self.nr_frames, 0, 3)
        layout.addWidget(self.show_mean_btn, 1, 0)
        layout.addWidget(self.slider, 1, 1)
        layout.addWidget(frame_label, 1, 2)
        layout.addWidget(self.frame_nr, 1, 3)
        layout.addWidget(imageWidget, 2, 0, 1, 4)
        layout.addWidget(self.actionBtns, 0, 4)
        layout.addWidget(DF_Widget, 0, 5, 3, 1)

    def setData(self, in_dataObj):
        self.dataObj = in_dataObj
        self.mean_data = np.array(np.mean(self.dataObj.data, 0), dtype=np.float32)
        self.show_mean()
        self.data_frames = self.dataObj.frames
        self.nr_frames.setText(str(self.data_frames))
        self.data_name.setText(self.dataObj.name)
        self.slider.setMaximum(self.data_frames - 1)


    def slider_moved(self):
        self.frame_nr.setText(str(self.slider.value()))
        self.setImgSlice()

    def setImgSlice(self):
        try:
            i = int(self.frame_nr.text())
        except TypeError:
            print('ERROR: Input must be an integer value')

        self.slider.setValue(i)
        self.img.setImage(self.dataObj.data[i], autoLevels=False)
        self.frame_nr.setText(str(i))

    def show_mean(self):
        self.img.setImage(self.mean_data)

    def setDarkFrame(self):
#        self.dataObj.data = self.dataObj.data[0:100]
        pass

class ScanningParWindow(QtGui.QMainWindow):
    """Seperate window for editing scanning parameters"""
    def __init__(self, main, parDict, r_l_text, u_d_text, b_f_text, timepoints_text, p_text, n_text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parDict = parDict

        im_dim_label = QtGui.QLabel('Image dimension')
        dim_dir_label = QtGui.QLabel('Direction')
        im_steps_label = QtGui.QLabel('Steps')
        im_step_size_label = QtGui.QLabel('Step size (nm)')

        self.r_l_text = r_l_text
        self.u_d_text = u_d_text
        self.b_f_text = b_f_text
        self.p_text = p_text
        self.timepoints_text = timepoints_text
        self.n_text = n_text

        dim0_label = QtGui.QLabel('Dimension 0')
        self.dim0_dim_edit = QtGui.QComboBox()
        self.dim0_dim_edit.addItems([self.r_l_text, self.u_d_text, self.b_f_text])
        self.dim0_dim_edit.currentIndexChanged.connect(self.dim0_changed)
        self.dim0_dir_edit = QtGui.QComboBox()
        self.dim0_dir_edit.addItems([self.p_text, self.n_text])
        self.dim0_size_edit = QtGui.QLineEdit()
        self.dim0_size_edit.returnPressed.connect(lambda: self.checkForInt(self.dim0_size_edit))
        self.dim0_step_size_edit = QtGui.QLineEdit()

        dim1_label = QtGui.QLabel('Dimension 1')
        self.dim1_dim_edit = QtGui.QComboBox()
        self.dim1_dim_edit.currentIndexChanged.connect(self.dim1_changed)
        self.dim1_dir_edit = QtGui.QComboBox()
        self.dim1_dir_edit.addItems([self.p_text, self.n_text])
        self.dim1_size_edit = QtGui.QLineEdit()
        self.dim1_size_edit.returnPressed.connect(lambda: self.checkForInt(self.dim1_size_edit))
        self.dim1_step_size_edit = QtGui.QLineEdit()

        dim2_label = QtGui.QLabel('Dimension 2')
        self.dim2_dim_edit = QtGui.QComboBox()
        self.dim2_dir_edit = QtGui.QComboBox()
        self.dim2_dir_edit.addItems([self.p_text, self.n_text])
        self.dim2_size_edit = QtGui.QLineEdit()
        self.dim2_size_edit.returnPressed.connect(lambda: self.checkForInt(self.dim2_size_edit))
        self.dim2_step_size_edit = QtGui.QLineEdit()

        dim3_label = QtGui.QLabel('Dimension 3')
        self.dim3_dim_label = QtGui.QLabel(self.timepoints_text)
        self.dim3_size_edit = QtGui.QLineEdit()
        self.dim3_size_edit.returnPressed.connect(lambda: self.checkForInt(self.dim3_size_edit))
        self.dim3_step_size_edit = QtGui.QLineEdit()

        self.unidir_check = QtGui.QCheckBox('Unidirectional scan')

        OK_btn = QtGui.QPushButton('OK')
        OK_btn.pressed.connect(self.OK_pressed)

        layout = QtGui.QGridLayout()
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        self.cwidget.setLayout(layout)

        layout.addWidget(im_dim_label, 0, 1)
        layout.addWidget(dim_dir_label, 0, 2)
        layout.addWidget(im_steps_label, 0, 3)
        layout.addWidget(im_step_size_label, 0, 4)
        layout.addWidget(dim0_label, 1, 0)
        layout.addWidget(self.dim0_dim_edit, 1, 1)
        layout.addWidget(self.dim0_dir_edit, 1, 2)
        layout.addWidget(self.dim0_size_edit, 1, 3)
        layout.addWidget(self.dim0_step_size_edit, 1, 4)
        layout.addWidget(dim1_label, 2, 0)
        layout.addWidget(self.dim1_dim_edit, 2, 1)
        layout.addWidget(self.dim1_dir_edit, 2, 2)
        layout.addWidget(self.dim1_size_edit, 2, 3)
        layout.addWidget(self.dim1_step_size_edit, 2, 4)
        layout.addWidget(dim2_label, 3, 0)
        layout.addWidget(self.dim2_dim_edit, 3, 1)
        layout.addWidget(self.dim2_dir_edit, 3, 2)
        layout.addWidget(self.dim2_size_edit, 3, 3)
        layout.addWidget(self.dim2_step_size_edit, 3, 4)
        layout.addWidget(dim3_label, 4, 0)
        layout.addWidget(self.dim3_dim_label, 4, 1)
        layout.addWidget(self.dim3_size_edit, 4, 3)
        layout.addWidget(self.dim3_step_size_edit, 4, 4)
        layout.addWidget(self.unidir_check, 5, 1)
        layout.addWidget(OK_btn, 5, 2)

        #Initiate values
        try:
            self.dim0_dim_edit.setCurrentIndex(self.dim0_dim_edit.findText(self.parDict['dimensions'][0]))
            self.dim1_dim_edit.setCurrentIndex(self.dim1_dim_edit.findText(self.parDict['dimensions'][1]))
            self.dim2_dim_edit.setCurrentIndex(self.dim2_dim_edit.findText(self.parDict['dimensions'][2]))
            self.dim0_changed()


            self.dim0_dir_edit.setCurrentIndex(self.dim0_dir_edit.findText(self.parDict['directions'][0]))
            self.dim1_dir_edit.setCurrentIndex(self.dim1_dir_edit.findText(self.parDict['directions'][1]))
            self.dim2_dir_edit.setCurrentIndex(self.dim2_dir_edit.findText(self.parDict['directions'][2]))

            self.dim0_size_edit.setText(self.parDict['steps'][0])
            self.dim1_size_edit.setText(self.parDict['steps'][1])
            self.dim2_size_edit.setText(self.parDict['steps'][2])
            self.dim3_size_edit.setText(self.parDict['steps'][3])

            self.dim0_step_size_edit.setText(self.parDict['step_sizes'][0])
            self.dim1_step_size_edit.setText(self.parDict['step_sizes'][1])
            self.dim2_step_size_edit.setText(self.parDict['step_sizes'][2])
            self.dim3_step_size_edit.setText(self.parDict['step_sizes'][3])

            self.unidir_check.setChecked(self.parDict['unidirectional'])
        except:
            print('Error setting initial values')
            self.dim0_changed()

    def checkForInt(self, parameter):
        try:
            int(parameter.text())
        except ValueError:
            parameter.setText('1')
            print('Cannit interpret given value as integer')

    def dim0_changed(self):
        currText = self.dim0_dim_edit.currentText()
        self.dim1_dim_edit.clear()
        if currText == self.r_l_text:
            self.dim1_dim_edit.addItems([self.u_d_text, self.b_f_text])
        elif currText == self.u_d_text:
            self.dim1_dim_edit.addItems([self.r_l_text, self.b_f_text])
        else:
            self.dim1_dim_edit.addItems([self.r_l_text, self.u_d_text])

        self.dim1_changed()

    def dim1_changed(self):
        currdim0Text = self.dim0_dim_edit.currentText()
        currdim1Text = self.dim1_dim_edit.currentText()
        self.dim2_dim_edit.clear()
        if currdim0Text == self.r_l_text:
            if currdim1Text == self.u_d_text:
                self.dim2_dim_edit.addItem(self.b_f_text)
            else:
                self.dim2_dim_edit.addItem(self.u_d_text)
        elif currdim0Text == self.u_d_text:
            if currdim1Text == self.r_l_text:
                self.dim2_dim_edit.addItem(self.b_f_text)
            else:
                self.dim2_dim_edit.addItem(self.r_l_text)
        else:
            if currdim1Text == self.r_l_text:
                self.dim2_dim_edit.addItem(self.u_d_text)
            else:
                self.dim2_dim_edit.addItem(self.r_l_text)

    def OK_pressed(self):

        self.parDict['dimensions'] = [self.dim0_dim_edit.currentText(),
                                        self.dim1_dim_edit.currentText(),
                                        self.dim2_dim_edit.currentText(),
                                        self.dim3_dim_label.text()]

        self.parDict['directions'] = [self.dim0_dir_edit.currentText(),
                                        self.dim1_dir_edit.currentText(),
                                        self.dim2_dir_edit.currentText(),
                                        self.p_text]

        self.parDict['steps'] = [self.dim0_size_edit.text(),
                                self.dim1_size_edit.text(),
                                self.dim2_size_edit.text(),
                                self.dim3_size_edit.text()]

        self.parDict['step_sizes'] = [self.dim0_step_size_edit.text(),
                                self.dim1_step_size_edit.text(),
                                self.dim2_step_size_edit.text(),
                                self.dim3_step_size_edit.text()]

        self.parDict['unidirectional'] = self.unidir_check.isChecked()

        self.close()


class ReconObj(object):
    def __init__(self, name, scanningParDict, r_l_text, u_d_text, b_f_text, timepoints_text, p_text, n_text, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.r_l_text = r_l_text
        self.u_d_text = u_d_text
        self.b_f_text = b_f_text
        self.timepoints_text = timepoints_text
        self.p_text = p_text
        self.n_tetx = n_text

        self.name = name
        self.coeffs = None
        self.reconstructed = None
        self.scanningParDict = scanningParDict.copy()

        self.disp_levels = None

    def setDispLevels(self, levels):
        self.disp_levels = levels

    def getDispLevels(self):
        return self.disp_levels

    def getReconstruction(self):
        return self.reconstructed

    def getCoeffs(self):
        return self.coeffs

    def addCoeffsTP(self, in_coeffs):
        if self.coeffs is None:
#            print('In if, shape is: ', np.shape(in_coeffs))
#            print('coeffs are: ', in_coeffs)
            self.coeffs = np.array([in_coeffs])
        else:
#            print('In else, shape self.data is: ', np.shape(in_coeffs))
#            print('In else, shape in_coeffs is: ', np.shape(in_coeffs))
            print('Max in coeffs = ', in_coeffs.max())
            in_coeffs = np.expand_dims(in_coeffs, 0)
            self.coeffs = np.vstack((self.coeffs, in_coeffs))

    def updateScanningPars(self, scanningParDict):
        self.scanningParDict = scanningParDict

    def update_images(self):
        """Updates the variable self.reconstructed which contains the final
        reconstructed and reassigned images of ALL the bases given to the
        reconstructor"""
        if not self.coeffs is None:
            datasets = np.shape(self.coeffs)[0]
            bases = np.shape(self.coeffs)[1]
            self.reconstructed = np.array([[self.coeffs_to_image(self.coeffs[ds][b], self.scanningParDict) for b in range(0, bases)] for ds in range(0, datasets)])
            print('shape of reconstructed is : ', np.shape(self.reconstructed))
        else:
            print('Cannot update images without coefficients')

    def add_grid_of_coeffs(self, im, coeffs, t, s, r0, c0, pr, pc):
#        print('Timepoint: ', t)
#        print('shape if im: ', im.shape)
#        print('shape if coeffts[i]: ', coeffs.shape)
#        print('r0: ', r0)
#        print('c0: ', c0)
#        print('pr: ', pr)
#        print('pc: ', pc)
        im[t, s, r0::pr, c0::pc] = coeffs

    def coeffs_to_image(self, coeffs, scanningParDict):
        """Takes the 4d matrix of coefficients from the signal extraction and
        reshapes into images according to given parameters"""
        frames = np.shape(coeffs)[0]
        dim0_side = int(scanningParDict['steps'][0])
        dim1_side = int(scanningParDict['steps'][1])
        dim2_side = int(scanningParDict['steps'][2])
        dim3_side = int(scanningParDict['steps'][3]) #Always timepoints
        if not frames == dim0_side*dim1_side*dim2_side*dim3_side:
            print('ERROR: Wrong dimensional data')
            pass

        timepoints = int(scanningParDict['steps'][scanningParDict['dimensions'].index(self.timepoints_text)])
        slices = int(scanningParDict['steps'][scanningParDict['dimensions'].index(self.b_f_text)])
        sq_rows = int(scanningParDict['steps'][scanningParDict['dimensions'].index(self.u_d_text)])
        sq_cols = int(scanningParDict['steps'][scanningParDict['dimensions'].index(self.r_l_text)])

        im = np.zeros([timepoints, slices, sq_rows*np.shape(coeffs)[1], sq_cols*np.shape(coeffs)[2]], dtype=np.float32)
        for i in np.arange(np.shape(coeffs)[0]):

            t = int(np.floor(i/(frames/dim3_side)))

            slow = int(np.mod(i, frames/timepoints) / (dim0_side*dim1_side))
            mid = int(np.mod(i, dim0_side*dim1_side) / dim0_side)
            fast = np.mod(i, dim0_side)

            if not scanningParDict['unidirectional']:
                odd_mid_step = np.mod(mid, 2)
                fast = (1-odd_mid_step)*fast + odd_mid_step*(dim1_side - 1 - fast)

            neg = (int(scanningParDict['directions'][0] == 'neg'),
                   int(scanningParDict['directions'][1] == 'neg'),
                   int(scanningParDict['directions'][2] == 'neg'))

            """Adjust for positive or negative direction"""
            fast = (1-neg[0])*fast + neg[0]*(dim0_side - 1 - fast)
            mid = (1-neg[1])*mid + neg[1]*(dim1_side - 1 - mid)
            slow = (1-neg[2])*slow + neg[2]*(dim2_side - 1 - slow)

            """Place dimensions in correct row/col/slice"""
            if scanningParDict['dimensions'][0] == self.r_l_text:
                if scanningParDict['dimensions'][1] == self.u_d_text:
                    c = fast
                    pc = dim0_side
                    r = mid
                    pr = dim1_side
                    s = slow
                else:
                    c = fast
                    pc = dim0_side
                    r = slow
                    pr = dim2_side
                    s = mid
            elif scanningParDict['dimensions'][0] == self.u_d_text:
                if scanningParDict['dimensions'][1] == self.r_l_text:
                    c = mid
                    pc = dim1_side
                    r = fast
                    pr = dim0_side
                    s = slow
                else:
                    c = slow
                    pc = dim2_side
                    r = fast
                    pr = dim0_side
                    s = mid
            else:
                if scanningParDict['dimensions'][1] == self.r_l_text:
                    c = mid
                    pc = dim1_side
                    r = slow
                    pr = dim2_side
                    s = fast
                else:
                    c = slow
                    pc = dim2_side
                    r = mid
                    pr = dim1_side
                    s = fast

            self.add_grid_of_coeffs(im, coeffs[i], t, s, r, c, pr, pc)

        return im


class DataObj(object):
    def __init__(self, name, path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_loaded = False
        self.name = name
        self.data_path = path
        self.data = None
        self.dark_frame = None
        self.frames = None

    def checkAndLoadData(self):

        if self.data_loaded:
            pass
        else:
            try:
                self.data = DataIO_tools.load_data(self.data_path, dtype=np.uint16)
                if not self.data is None:
                    print('Data loaded')
                    self.data_loaded = True
                    self.frames = np.shape(self.data)[0]
            except:
                pass

    def checkAndLoadDarkFrame(self):
        pass

    def checkAndUnloadData(self):

        try:
            if self.data_loaded:
                self.data = None
            else:
                pass
        except:
            print('Error while unloading data')

        self.data_loaded = False


def show_im_seq(seq):
    for i in np.arange(seq.shape[0]):
        plt.imshow(seq[i], 'gray')
        plt.pause(0.01)

def load_data(path):
    path = os.path.abspath(path)
    try:
        ext = os.path.splitext(path)[1]

        if ext in ['.hdf5', '.hdf']:
            with h5py.File(path, 'r') as datafile:
                data = np.array(datafile['Images'][:])

        elif ext in ['.tiff', '.tif']:
            with tiff.TiffFile(path) as datafile:
                data = datafile.asarray()

        return data
    except:
        print('Error while loading data')
        return None

if __name__ == "__main__":

    app = QtGui.QApplication([])

    wid = ReconWid()
    wid.show()

    sys.exit(app.exec_())
