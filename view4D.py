import pyqtgraph as pg
import numpy as np
from IPython.display import display

import jupyter_rfb
import numpy as np

from pyqtgraph import functions as fn
from pyqtgraph import graphicsItems, widgets
from pyqtgraph.Qt import QtCore, QtGui

import h5py
import xarray as xr

# Created by Maciej Dendzik 2024

__all__ = ['GraphicsLayoutWidget', 'PlotWidget','View4D']

KMLUT = {
    x : getattr(QtCore.Qt.KeyboardModifier, x + "Modifier")
    for x in ["Shift", "Control", "Alt", "Meta"]
}

MBLUT = {
    k : getattr(QtCore.Qt.MouseButton, v + "Button")
    for (k, v) in zip(
        range(6),
        ["No", "Left", "Right", "Middle", "Back", "Forward"]
    )
}

TYPLUT = {
    "pointer_down" : QtCore.QEvent.Type.MouseButtonPress,
    "pointer_up" : QtCore.QEvent.Type.MouseButtonRelease,
    "pointer_move" : QtCore.QEvent.Type.MouseMove,
}

KEYLUT = {
    'a': QtCore.Qt.Key_A,
    'b': QtCore.Qt.Key_B,
    'c': QtCore.Qt.Key_C,
    'd': QtCore.Qt.Key_D,
    'e': QtCore.Qt.Key_E,
    'f': QtCore.Qt.Key_F,
    'g': QtCore.Qt.Key_G,
    'h': QtCore.Qt.Key_H,
    'i': QtCore.Qt.Key_I,
    'j': QtCore.Qt.Key_J,
    'k': QtCore.Qt.Key_K,
    'l': QtCore.Qt.Key_L,
    'm': QtCore.Qt.Key_M,
    'n': QtCore.Qt.Key_N,
    'o': QtCore.Qt.Key_O,
    'p': QtCore.Qt.Key_P,
    'q': QtCore.Qt.Key_Q,
    'r': QtCore.Qt.Key_R,
    's': QtCore.Qt.Key_S,
    't': QtCore.Qt.Key_T,
    'u': QtCore.Qt.Key_U,
    'v': QtCore.Qt.Key_V,
    'w': QtCore.Qt.Key_W,
    'x': QtCore.Qt.Key_X,
    'y': QtCore.Qt.Key_Y,
    'z': QtCore.Qt.Key_Z,
    
    'A': QtCore.Qt.Key_A,
    'B': QtCore.Qt.Key_B,
    'C': QtCore.Qt.Key_C,
    'D': QtCore.Qt.Key_D,
    'E': QtCore.Qt.Key_E,
    'F': QtCore.Qt.Key_F,
    'G': QtCore.Qt.Key_G,
    'H': QtCore.Qt.Key_H,
    'I': QtCore.Qt.Key_I,
    'J': QtCore.Qt.Key_J,
    'K': QtCore.Qt.Key_K,
    'L': QtCore.Qt.Key_L,
    'M': QtCore.Qt.Key_M,
    'N': QtCore.Qt.Key_N,
    'O': QtCore.Qt.Key_O,
    'P': QtCore.Qt.Key_P,
    'Q': QtCore.Qt.Key_Q,
    'R': QtCore.Qt.Key_R,
    'S': QtCore.Qt.Key_S,
    'T': QtCore.Qt.Key_T,
    'U': QtCore.Qt.Key_U,
    'V': QtCore.Qt.Key_V,
    'W': QtCore.Qt.Key_W,
    'X': QtCore.Qt.Key_X,
    'Y': QtCore.Qt.Key_Y,
    'Z': QtCore.Qt.Key_Z,
    
    '0': QtCore.Qt.Key_0,
    '1': QtCore.Qt.Key_1,
    '2': QtCore.Qt.Key_2,
    '3': QtCore.Qt.Key_3,
    '4': QtCore.Qt.Key_4,
    '5': QtCore.Qt.Key_5,
    '6': QtCore.Qt.Key_6,
    '7': QtCore.Qt.Key_7,
    '8': QtCore.Qt.Key_8,
    '9': QtCore.Qt.Key_9,
    
    'Space': QtCore.Qt.Key_Space,
    'Enter': QtCore.Qt.Key_Return,
    'Return': QtCore.Qt.Key_Return,
    'Escape': QtCore.Qt.Key_Escape,
    'Backspace': QtCore.Qt.Key_Backspace,
    'Delete': QtCore.Qt.Key_Delete,
    'Tab': QtCore.Qt.Key_Tab,
    'Shift': QtCore.Qt.Key_Shift,
    'Control': QtCore.Qt.Key_Control,
    'Alt': QtCore.Qt.Key_Alt,
    'Meta': QtCore.Qt.Key_Meta,    # Windows key or Command key on macOS
    'CapsLock': QtCore.Qt.Key_CapsLock,
    'ArrowUp': QtCore.Qt.Key_Up,
    'ArrowDown': QtCore.Qt.Key_Down,
    'ArrowLeft': QtCore.Qt.Key_Left,
    'ArrowRight': QtCore.Qt.Key_Right,
    
    'F1': QtCore.Qt.Key_F1,
    'F2': QtCore.Qt.Key_F2,
    'F3': QtCore.Qt.Key_F3,
    'F4': QtCore.Qt.Key_F4,
    'F5': QtCore.Qt.Key_F5,
    'F6': QtCore.Qt.Key_F6,
    'F7': QtCore.Qt.Key_F7,
    'F8': QtCore.Qt.Key_F8,
    'F9': QtCore.Qt.Key_F9,
    'F10': QtCore.Qt.Key_F10,
    'F11': QtCore.Qt.Key_F11,
    'F12': QtCore.Qt.Key_F12,
    
    'Comma': QtCore.Qt.Key_Comma,
    'Period': QtCore.Qt.Key_Period,
    'Slash': QtCore.Qt.Key_Slash,
    'Semicolon': QtCore.Qt.Key_Semicolon,
    'Apostrophe': QtCore.Qt.Key_Apostrophe,
    'BracketLeft': QtCore.Qt.Key_BracketLeft,
    'BracketRight': QtCore.Qt.Key_BracketRight,
    'Minus': QtCore.Qt.Key_Minus,
    'Equal': QtCore.Qt.Key_Equal,
    'Backslash': QtCore.Qt.Key_Backslash,
    'Grave': QtCore.Qt.Key_QuoteLeft,  # Also known as backtick (`) key
}

def get_buttons(evt_buttons):
    NoButton = QtCore.Qt.MouseButton.NoButton
    btns = NoButton
    for x in evt_buttons:
        btns |= MBLUT.get(x, NoButton)
    return btns

def get_modifiers(evt_modifiers):
    NoModifier = QtCore.Qt.KeyboardModifier.NoModifier
    mods = NoModifier
    for x in evt_modifiers:
        mods |= KMLUT.get(x, NoModifier)
    return mods


class GraphicsView(jupyter_rfb.RemoteFrameBuffer):
    """jupyter_rfb.RemoteFrameBuffer sub-class that wraps around
    :class:`GraphicsView <pyqtgraph.GraphicsView>`.

    Generally speaking, there is no Qt event loop running. The implementation works by
    requesting a render() of the scene. Thus things that would work for exporting
    purposes would be expected to work here. Things that are not part of the scene
    would not work, e.g. context menus, tooltips.

    This class should not be used directly. Its corresponding sub-classes
    :class:`GraphicsLayoutWidget <pyqtgraph.jupyter.GraphicsLayoutWidget>` and
    :class:`PlotWidget <pyqtgraph.jupyter.PlotWidget>` should be used instead."""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.gfxView = widgets.GraphicsView.GraphicsView()
        self.logical_size = int(self.css_width[:-2]), int(self.css_height[:-2])
        self.pixel_ratio = 1.0
        # self.gfxView.resize(*self.logical_size)
        # self.gfxView.show()
        # self.gfxView.resizeEvent(None)

    def get_frame(self):
        w, h = self.logical_size
        dpr = self.pixel_ratio
        buf = np.empty((int(h * dpr), int(w * dpr), 4), dtype=np.uint8)
        qimg = fn.ndarray_to_qimage(buf, QtGui.QImage.Format.Format_RGBX8888)
        qimg.fill(QtCore.Qt.GlobalColor.transparent)
        qimg.setDevicePixelRatio(dpr)
        painter = QtGui.QPainter(qimg)
        self.gfxView.render(painter, self.gfxView.viewRect(), self.gfxView.rect())
        painter.end()
        return buf
    
    def handle_event(self, event):
        event_type = event["event_type"]

        if event_type == "resize":
            oldSize = QtCore.QSize(*self.logical_size)
            self.logical_size = int(event["width"]), int(event["height"])
            self.pixel_ratio = event["pixel_ratio"]
            self.gfxView.resize(*self.logical_size)
            newSize = QtCore.QSize(*self.logical_size)
            self.gfxView.resizeEvent(QtGui.QResizeEvent(newSize, oldSize))
        elif event_type in ["pointer_down", "pointer_up", "pointer_move"]:
            btn = MBLUT.get(event["button"], None)
            if btn is None:    # ignore unsupported buttons
                return
            pos = QtCore.QPointF(event["x"], event["y"])
            btns = get_buttons(event["buttons"])
            mods = get_modifiers(event["modifiers"])
            typ = TYPLUT[event_type]
            evt = QtGui.QMouseEvent(typ, pos, pos, btn, btns, mods)
            QtCore.QCoreApplication.sendEvent(self.gfxView.viewport(), evt)
            self.request_draw()
        elif event_type == "wheel":
            pos = QtCore.QPointF(event["x"], event["y"])
            pixdel = QtCore.QPoint()
            scale = -1.0    # map JavaScript wheel to Qt wheel
            angdel = QtCore.QPoint(int(event["dx"] * scale), int(event["dy"] * scale))
            btns = get_buttons([])
            mods = get_modifiers(event["modifiers"])
            phase = QtCore.Qt.ScrollPhase.NoScrollPhase
            inverted = False
            evt = QtGui.QWheelEvent(pos, pos, pixdel, angdel, btns, mods, phase, inverted)
            QtCore.QCoreApplication.sendEvent(self.gfxView.viewport(), evt)
        elif event_type == "key_down":
            typ=QtCore.QEvent.KeyPress
            mods = get_modifiers(event["modifiers"])
            key=KEYLUT[event['key']]
            evt=QtGui.QKeyEvent(typ, key, mods)
            QtCore.QCoreApplication.sendEvent(self.gfxView.viewport(), evt)
            


def connect_viewbox_redraw(vb, request_draw):
    # connecting these signals is enough to support zoom/pan
    # but not enough to support the various graphicsItems
    # that react to mouse events

    vb.sigRangeChanged.connect(request_draw)
    # zoom / pan
    vb.sigRangeChangedManually.connect(request_draw)
    # needed for "auto" button
    vb.sigStateChanged.connect(request_draw)
    # note that all cases of sig{X,Y}RangeChanged being emitted
    # are also followed by sigRangeChanged or sigStateChanged
    vb.sigTransformChanged.connect(request_draw)


class GraphicsLayoutWidget(GraphicsView):
    """jupyter_rfb analogue of
    :class:`GraphicsLayoutWidget <pyqtgraph.GraphicsLayoutWidget>`."""

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.gfxLayout = graphicsItems.GraphicsLayout.GraphicsLayout()
        for n in [
            'nextRow', 'nextCol', 'nextColumn', 'addItem', 'getItem',
            'addLayout', 'addLabel', 'removeItem', 'itemIndex', 'clear'
        ]:
            setattr(self, n, getattr(self.gfxLayout, n))
        self.gfxView.setCentralItem(self.gfxLayout)

    def addPlot(self, *args, **kwds):
        kwds["enableMenu"] = False
        plotItem = self.gfxLayout.addPlot(*args, **kwds)
        connect_viewbox_redraw(plotItem.getViewBox(), self.request_draw)
        return plotItem

    def addViewBox(self, *args, **kwds):
        kwds["enableMenu"] = False
        vb = self.gfxLayout.addViewBox(*args, **kwds)
        connect_viewbox_redraw(vb, self.request_draw)
        return vb


class PlotWidget(GraphicsView):
    """jupyter_rfb analogue of
    :class:`PlotWidget <pyqtgraph.PlotWidget>`."""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        plotItem = graphicsItems.PlotItem.PlotItem(enableMenu=False)
        self.gfxView.setCentralItem(plotItem)
        connect_viewbox_redraw(plotItem.getViewBox(), self.request_draw)
        self.plotItem = plotItem

    def getPlotItem(self):
        return self.plotItem

    def __getattr__(self, attr):
        # kernel crashes if we don't skip attributes starting with '_'
        if attr.startswith('_'):
            return super().__getattr__(attr)

        # implicitly wrap methods from plotItem
        if hasattr(self.plotItem, attr):
            m = getattr(self.plotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise AttributeError(attr)

		
		
class View4D:
    def __init__(self, xr_data, color_map='inferno'):
        self.xr_data = xr_data
        self.color_map = color_map
        # self.cmap = pg.colormap.get('viridis')
        
        # Set up PyQtGraph configurations
        pg.setConfigOptions(imageAxisOrder='col-major')
        pg.setConfigOption('useNumba', True)
        pg.mkQApp()
        
        # Set up the main window and plot layout
        self.win = GraphicsLayoutWidget(css_width="1000px", css_height="1000px")
        
        
        self.x0, self.y0, self.z0, self.t0 = [float(xr_data.coords[dim][0].values) for dim in xr_data.dims]
        self.xs, self.ys, self.zs, self.ts = [float(xr_data.coords[dim][1].values) - float(xr_data.coords[dim][0].values) for dim in xr_data.dims]
        self.xend, self.yend, self.zend, self.tend = [float(xr_data.coords[dim][-1].values) for dim in xr_data.dims]
        
        # First row
        self.init_main_plot()
        self.init_xz_plot()
        # Second row
        self.win.nextRow()
        self.init_main_zcut()
        self.init_xz_zcut()
        #Third row
        self.win.nextRow()
        self.init_main_tcut()
        self.init_xz_xcut()
        #Fourth row
        self.win.nextRow()
        self.init_yz_plot()
        self.init_zt_plot()
        #Fith row
        self.win.nextRow()
        self.init_yz_zcut()
        self.init_zt_zcut()
        #Sixth row
        self.win.nextRow()
        self.init_yz_ycut()
        self.init_zt_tcut()
        
        self.p1view.keyPressEvent = self.keypressimg
        self.p1xzview.keyPressEvent = self.keypressimgxz
        self.p1yzview.keyPressEvent = self.keypressimgyz
        self.p1ztview.keyPressEvent = self.keypressimgzt


        self.p2view.keyPressEvent = self.keypress
        self.p3view.keyPressEvent = self.keypress

        self.p2viewxz.keyPressEvent = self.keypressxz
        self.p3viewxz.keyPressEvent = self.keypressxz

        self.p2viewyz.keyPressEvent = self.keypressyz
        self.p3viewyz.keyPressEvent = self.keypressyz

        self.p2viewzt.keyPressEvent = self.keypresszt
        self.p3viewzt.keyPressEvent = self.keypresszt
        
       
        
        # Display the window
        display(self.win)
	
    def close(self):
        self.win.close()
		

        
    def init_main_plot(self):
        self.p1 = self.win.addPlot(title="")
        self.p1view=self.p1.getViewBox()
        # Item for displaying image data
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        
        # limits = (0., self.xr_data.sum(axis=(2, 3)).values.max())
        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        tr.translate(self.x0, self.y0) # 
        tr.scale(self.xs,self.ys)       # scale horizontal and vertical axes
        self.p1.setLabel("bottom", self.xr_data.dims[0])  # Set x-axis label
        self.p1.setLabel("left", self.xr_data.dims[1])  # Set y-axis label
        self.img.setTransform(tr) # assign transform
        data = self.xr_data.sum(axis=(2,3)).values
        self.img.setImage(data)

        limits = 0., data.max()
        bar = pg.ColorBarItem(values=limits, colorMap=self.color_map, limits=limits, rounding=.1,width=15)
        bar.setImageItem(self.img, insert_in=self.p1)
  
        # # Custom ROI for selecting an image region
        roi_bounds = QtCore.QRectF(self.x0,self.y0,self.xend-self.x0,self.yend-self.y0)


        # roi = pg.ROI([x0,y0], [xend-x0,-y0+yend],maxBounds=roi_bounds)

        self.roi = pg.ROI([(self.xend+self.x0)/2-(self.xend-self.x0)/8,(self.yend+self.y0)/2-(-self.y0+self.yend)/8], [(self.xend-self.x0)/4,(-self.y0+self.yend)/4],maxBounds=roi_bounds)
        self.roi.addScaleHandle([1, 1], [0.5, 0.5])
        self.roi.addTranslateHandle([0.5, 0.5])
        self.roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        # roi.addScaleHandle([0, 0.5], [0.5, 0.5])

        self.roi2 = pg.ROI(pos=self.roi.pos(),size=(0,0),maxBounds=roi_bounds,movable=False,pen=(0,255,0),removable=True)
        self.p1.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image
        self.p1.addItem(self.roi2)
        self.roi2.setZValue(5)  # make sure ROI is drawn above image
        
        self.roi.sigRegionChanged.connect(self.updatePlot)
        self.roi.sigClicked.connect(self.roi1Clicked)
        self.roi.sigRegionChangeFinished.connect(self.updatePlotFinished)
        
    def roi1Clicked(self,roi,event):
        self.img.setLevels((np.min(roi.getArrayRegion(self.img.image,self.img)),np.max(roi.getArrayRegion(self.img.image,self.img))))
    
    def updatePlot(self):
        selected = self.roi.getState()
        self.p2line.setData(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values)
        self.p3line.setData(self.xr_data.coords[self.xr_data.dims[3]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[2]:slice(*self.zRegion.getRegion())}].sum(axis=(0,1,2)).values)
    
    def updatePlotFinished(self):
        selected = self.roi.getState()
        selectedxz = self.roixz.getState()
        posz=self.zRegion.getRegion()
        post=self.tRegion.getRegion()
        self.imgxz.setImage(self.xr_data.loc[{self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(1,3)).values,autoLevels=False)
        self.imgyz.setImage(self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,3)).values,autoLevels=False)
        self.imgzt.setImage(self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1])}].sum(axis=(0,1)).T.values,autoLevels=False)



        self.p1.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[3]} = {post[0]:.3f} + {post[1] - post[0]:.3f}  <br>"
        f"{self.xr_data.dims[0]} = {selected['pos'][0] + selected['size'][0] / 2:.3f} <span>&#177;</span> {selected['size'][0] / 2:.3f}  "
        f" | {self.xr_data.dims[1]} = {selected['pos'][1] + selected['size'][1] / 2:.3f} <span>&#177;</span> {selected['size'][1] / 2:.3f}",
        wrap=True
        )
        
    def init_xz_plot(self):
        self.p1xz = self.win.addPlot(title="")
        self.p1xzview=self.p1xz.getViewBox()
        # Item for displaying image data
        self.imgxz = pg.ImageItem()
        self.p1xz.addItem(self.imgxz)
        
        # limits = (0., self.xr_data.sum(axis=(2, 3)).values.max())
        trxz = QtGui.QTransform()  # prepare ImageItem transformation:
        trxz.translate(self.x0, self.z0) # 
        trxz.scale(self.xs,self.zs)       # scale horizontal and vertical axes
        self.p1xz.setLabel("bottom", self.xr_data.dims[0])  # Set x-axis label
        self.p1xz.setLabel("left", self.xr_data.dims[2])  # Set y-axis label
        self.imgxz.setTransform(trxz) # assign transform
        dataxz = self.xr_data.sum(axis=(1,3)).values
        self.imgxz.setImage(dataxz)

        limits = 0., dataxz.max()
        bar = pg.ColorBarItem(values=limits, colorMap=self.color_map, limits=limits, rounding=.1,width=15)
        bar.setImageItem(self.imgxz, insert_in=self.p1xz)
  
        # # Custom ROI for selecting an image region
        roi_bounds = QtCore.QRectF(self.x0,self.z0,self.xend-self.x0,self.zend-self.z0)


        # roi = pg.ROI([x0,y0], [xend-x0,-y0+yend],maxBounds=roi_bounds)

        self.roixz = pg.ROI([(self.xend+self.x0)/2-(self.xend-self.x0)/8,(self.zend+self.z0)/2-(-self.z0+self.zend)/8], [(self.xend-self.x0)/4,(-self.z0+self.zend)/4],maxBounds=roi_bounds)
        self.roixz.addScaleHandle([1, 1], [0.5, 0.5])
        self.roixz.addTranslateHandle([0.5, 0.5])
        self.roixz.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        # roi.addScaleHandle([0, 0.5], [0.5, 0.5])

        self.roi2xz = pg.ROI(pos=self.roixz.pos(),size=(0,0),maxBounds=roi_bounds,movable=False,pen=(0,255,0),removable=True)
        self.p1xz.addItem(self.roixz)
        self.roixz.setZValue(10)  # make sure ROI is drawn above image
        self.p1xz.addItem(self.roi2xz)
        self.roi2xz.setZValue(5)  # make sure ROI is drawn above image
        
        self.roixz.sigRegionChanged.connect(self.updatePlotxz)
        self.roixz.sigClicked.connect(self.roi1Clickedxz)
        self.roixz.sigRegionChangeFinished.connect(self.updatePlotFinishedxz)
        
    def roi1Clickedxz(self,roi,event):
        self.imgxz.setLevels((np.min(roi.getArrayRegion(self.imgxz.image,self.imgxz)),np.max(roi.getArrayRegion(self.imgxz.image,self.imgxz))))
    
    def updatePlotxz(self):
        selected = self.roi.getState()
        selectedxz = self.roixz.getState()
        self.p2linexz.setData(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selectedxz['pos'][0],selectedxz['pos'][0]+selectedxz['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values)
        self.p3linexz.setData(self.xr_data.coords[self.xr_data.dims[0]].values,self.xr_data.loc[{self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[2]:slice(selectedxz['pos'][1],selectedxz['pos'][1]+selectedxz['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(1,2,3)).values)
    
    def updatePlotFinishedxz(self):
        selectedxz = self.roixz.getState()
        posz=self.zRegionxz.getRegion()
        posx=self.xRegionxz.getRegion()
        self.p1xz.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[0]} = {posx[0]:.3f} + {posx[1] - posx[0]:.3f}  <br>"
        f"EDC = {selectedxz['pos'][0] + selectedxz['size'][0] / 2:.3f} <span>&#177;</span> {selectedxz['size'][0] / 2:.3f}  "
        f" | MDC = {selectedxz['pos'][1] + selectedxz['size'][1] / 2:.3f} <span>&#177;</span> {selectedxz['size'][1] / 2:.3f}",
        wrap=True
        )
        
        
    def init_main_zcut(self):
        
        self.p2 = self.win.addPlot(colspan=1)
        self.p2view=self.p2.getViewBox()
        # p2view.setMouseMode(p2view.RectMode)
        self.p2.setMaximumHeight(100)
        self.p2line=self.p2.plot(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.sum(axis=(0,1,3)).values, clear=True)
        self.p2line2=self.p2.plot()
        self.zRegion = pg.LinearRegionItem([self.z0, self.zend], orientation='vertical', movable=True,bounds=[self.z0, self.zend])
        self.p2.addItem(self.zRegion)

        self.p2.setLabel("bottom", self.xr_data.dims[2])  # Set x-axis label
        self.zRegion.sigRegionChangeFinished.connect(self.updatezRegion)
        
    def updatezRegion(self):
        selected = self.roi.getState()
        # selectedxz = self.roixz.getState()
        
        self.img.setImage(self.xr_data.loc[{self.xr_data.dims[2]:slice(*self.zRegion.getRegion()),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(2,3)).values,autoLevels=False)
        self.p3line.setData(self.xr_data.coords[self.xr_data.dims[3]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[2]:slice(*self.zRegion.getRegion())}].sum(axis=(0,1,2)).values)
        posz=self.zRegion.getRegion()
        post=self.tRegion.getRegion()
        self.p1.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[3]} = {post[0]:.3f} + {post[1] - post[0]:.3f}  <br>"
        f"{self.xr_data.dims[0]} = {selected['pos'][0] + selected['size'][0] / 2:.3f} <span>&#177;</span> {selected['size'][0] / 2:.3f}  "
        f" | {self.xr_data.dims[1]} = {selected['pos'][1] + selected['size'][1] / 2:.3f} <span>&#177;</span> {selected['size'][1] / 2:.3f}",
        wrap=True
        )
    def init_xz_zcut(self):
        self.p2xz = self.win.addPlot(colspan=1)
        self.p2viewxz=self.p2xz.getViewBox()
        # p2view.setMouseMode(p2view.RectMode)
        self.p2xz.setMaximumHeight(100)
        self.p2linexz=self.p2xz.plot(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.sum(axis=(0,1,3)).values, clear=True)
        self.p2line2xz=self.p2xz.plot()
        self.zRegionxz = pg.LinearRegionItem([self.z0, self.zend], orientation='vertical', movable=True,bounds=[self.z0, self.zend])
        self.p2xz.addItem(self.zRegionxz)
        self.p2xz.setLabel("bottom", self.xr_data.dims[2])  # Set x-axis label
        self.zRegionxz.sigRegionChangeFinished.connect(self.updateRegionxz)
    
    def updateRegionxz():
        posz=self.zRegionxz.getRegion()
        posx=self.xRegionxz.getRegion()
        selectedxz = self.roixz.getState()
        # p1xz.setTitle(xr_data.dims[2]+f" ({posz[0]:.3f}, {posz[1]-posz[0]:.3f}) "+xr_data.dims[1]+f" ({posx[0]:.3f}, {posx[1]-posx[0]:.3f}) "+"<br>EDC"+f" ({selectedxz['pos'][0]+selectedxz['size'][0]/2:.3f}, {selectedxz['size'][0]/2:.3f}) MDC"+f" ({selectedxz['pos'][1]+selectedxz['size'][1]/2:.3f}, {selectedxz['size'][1]/2:.3f})",wrap=True)
        self.p1xz.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[0]} = {posx[0]:.3f} + {posx[1] - posx[0]:.3f}  <br>"
        f"EDC = {selectedxz['pos'][0] + selectedxz['size'][0] / 2:.3f} <span>&#177;</span> {selectedxz['size'][0] / 2:.3f}  "
        f" | MDC = {selectedxz['pos'][1] + selectedxz['size'][1] / 2:.3f} <span>&#177;</span> {selectedxz['size'][1] / 2:.3f}",
        wrap=True
        )
        
    def init_main_tcut(self):
        self.p3 = self.win.addPlot(colspan=1)
        self.p3view=self.p3.getViewBox()
        self.p3.setMaximumHeight(100)
        self.p3line=self.p3.plot(self.xr_data.coords[self.xr_data.dims[3]].values,self.xr_data.sum(axis=(0,1,2)).values, clear=True)
        self.p3line2=self.p3.plot()
        self.tRegion = pg.LinearRegionItem([self.t0, self.tend], orientation='vertical', movable=True,bounds=[self.t0, self.tend])
        self.p3.addItem(self.tRegion)
        self.p3.setLabel("bottom", self.xr_data.dims[3])  # Set x-axis label
        self.tRegion.sigRegionChangeFinished.connect(self.updatetRegion)
        
    def updatetRegion(self):
        selected = self.roi.getState()
        selectedxz = self.roixz.getState()

        self.img.setImage(self.xr_data.loc[{self.xr_data.dims[2]:slice(*self.zRegion.getRegion()),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(2,3)).values,autoLevels=False)
        self.imgxz.setImage(self.xr_data.loc[{self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(1,3)).values,autoLevels=False)
        self.imgyz.setImage(self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,3)).values,autoLevels=False)
        
        self.p2line.setData(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values)
        self.updatePlotxz()
        self.updatePlotyz()
        # self.updatePlotzt()
        
        posz=self.zRegion.getRegion()
        post=self.tRegion.getRegion()
        self.p1.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[3]} = {post[0]:.3f} + {post[1] - post[0]:.3f}  <br>"
        f"{self.xr_data.dims[0]} = {selected['pos'][0] + selected['size'][0] / 2:.3f} <span>&#177;</span> {selected['size'][0] / 2:.3f}  "
        f" | {self.xr_data.dims[1]} = {selected['pos'][1] + selected['size'][1] / 2:.3f} <span>&#177;</span> {selected['size'][1] / 2:.3f}",
        wrap=True
        )
        

    def init_xz_xcut(self):
        self.p3xz = self.win.addPlot(colspan=1)
        self.p3xz.setMaximumHeight(100)
        self.p3viewxz=self.p3xz.getViewBox()
        self.p3linexz=self.p3xz.plot(self.xr_data.coords[self.xr_data.dims[0]].values,self.xr_data.sum(axis=(1,2,3)).values, clear=True)
        self.p3linexz2=self.p3xz.plot()
        self.xRegionxz = pg.LinearRegionItem([self.x0, self.xend], orientation='vertical', movable=True,bounds=[self.x0, self.xend])
        self.p3xz.addItem(self.xRegionxz)
        self.p3xz.setLabel("bottom", self.xr_data.dims[0])  # Set x-axis label
        self.xRegionxz.sigRegionChangeFinished.connect(self.updateRegionxz)

        
    def updateRegionxz(self):
        posz=self.zRegionxz.getRegion()
        posx=self.xRegionxz.getRegion()
        selectedxz = self.roixz.getState()
        
        self.p1xz.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[0]} = {posx[0]:.3f} + {posx[1] - posx[0]:.3f}  <br>"
        f"EDC = {selectedxz['pos'][0] + selectedxz['size'][0] / 2:.3f} <span>&#177;</span> {selectedxz['size'][0] / 2:.3f}  "
        f" | MDC = {selectedxz['pos'][1] + selectedxz['size'][1] / 2:.3f} <span>&#177;</span> {selectedxz['size'][1] / 2:.3f}",
        wrap=True
        )
        
    def init_yz_plot(self):
        self.p1yz = self.win.addPlot(title="")
        self.p1yzview=self.p1yz.getViewBox()
        # Item for displaying image data
        self.imgyz = pg.ImageItem()
        self.p1yz.addItem(self.imgyz)
        
        # limits = (0., self.xr_data.sum(axis=(2, 3)).values.max())
        tryz = QtGui.QTransform()  # prepare ImageItem transformation:
        tryz.translate(self.y0, self.z0) # 
        tryz.scale(self.ys,self.zs)       # scale horizontal and vertical axes
        self.p1yz.setLabel("bottom", self.xr_data.dims[1])  # Set x-axis label
        self.p1yz.setLabel("left", self.xr_data.dims[2])  # Set y-axis label
        self.imgyz.setTransform(tryz) # assign transform
        datayz = self.xr_data.sum(axis=(0,3)).values
        self.imgyz.setImage(datayz)

        limits = 0., datayz.max()
        bar = pg.ColorBarItem(values=limits, colorMap=self.color_map, limits=limits, rounding=.1,width=15)
        bar.setImageItem(self.imgyz, insert_in=self.p1yz)
  
        # # Custom ROI for selecting an image region
        roi_bounds = QtCore.QRectF(self.y0,self.z0,self.yend-self.y0,self.zend-self.z0)


        # roi = pg.ROI([y0,y0], [yend-y0,-y0+yend],maxBounds=roi_bounds)

        self.roiyz = pg.ROI([(self.yend+self.y0)/2-(self.yend-self.y0)/8,(self.zend+self.z0)/2-(-self.z0+self.zend)/8], [(self.yend-self.y0)/4,(-self.z0+self.zend)/4],maxBounds=roi_bounds)
        self.roiyz.addScaleHandle([1, 1], [0.5, 0.5])
        self.roiyz.addTranslateHandle([0.5, 0.5])
        self.roiyz.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        # roi.addScaleHandle([0, 0.5], [0.5, 0.5])

        self.roi2yz = pg.ROI(pos=self.roiyz.pos(),size=(0,0),maxBounds=roi_bounds,movable=False,pen=(0,255,0),removable=True)
        self.p1yz.addItem(self.roiyz)
        self.roiyz.setZValue(10)  # make sure ROI is drawn above image
        self.p1yz.addItem(self.roi2yz)
        self.roi2yz.setZValue(5)  # make sure ROI is drawn above image
        
        self.roiyz.sigRegionChanged.connect(self.updatePlotyz)
        self.roiyz.sigClicked.connect(self.roi1Clickedyz)
        self.roiyz.sigRegionChangeFinished.connect(self.updatePlotFinishedyz)
        
    def roi1Clickedyz(self,roi,event):
        self.imgyz.setLevels((np.min(roi.getArrayRegion(self.imgyz.image,self.imgyz)),np.max(roi.getArrayRegion(self.imgyz.image,self.imgyz))))
    
    def updatePlotyz(self):
        selected = self.roi.getState()
        selectedyz = self.roiyz.getState()
        self.p2lineyz.setData(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selectedyz['pos'][0],selectedyz['pos'][0]+selectedyz['size'][0]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values)
        self.p3lineyz.setData(self.xr_data.coords[self.xr_data.dims[1]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[2]:slice(selectedyz['pos'][1],selectedyz['pos'][1]+selectedyz['size'][1]),self.xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,2,3)).values)

    def updatePlotFinishedyz(self):
        selectedyz = self.roiyz.getState()
        posz=self.zRegionyz.getRegion()
        posy=self.yRegionyz.getRegion()
        self.p1yz.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[1]} = {posy[0]:.3f} + {posy[1] - posy[0]:.3f}  <br>"
        f"EDC = {selectedyz['pos'][0] + selectedyz['size'][0] / 2:.3f} <span>&#177;</span> {selectedyz['size'][0] / 2:.3f}  "
        f" | MDC = {selectedyz['pos'][1] + selectedyz['size'][1] / 2:.3f} <span>&#177;</span> {selectedyz['size'][1] / 2:.3f}",
        wrap=True
        )
        
    def init_zt_plot(self):
        self.p1zt = self.win.addPlot(title="")
        self.p1ztview=self.p1zt.getViewBox()
        # Item for displaying image data
        self.imgzt = pg.ImageItem()
        self.p1zt.addItem(self.imgzt)
        
        # limits = (0., self.xr_data.sum(axis=(2, 3)).values.max())
        trzt = QtGui.QTransform()  # prepare ImageItem transformation:
        trzt.translate(self.t0, self.z0) # 
        trzt.scale(self.ts,self.zs)       # scale horizontal and vertical axes
        self.p1zt.setLabel("bottom", self.xr_data.dims[3])  # Set x-axis label
        self.p1zt.setLabel("left", self.xr_data.dims[2])  # Set y-axis label
        self.imgzt.setTransform(trzt) # assign transform
        datazt = self.xr_data.sum(axis=(0,1)).T.values
        self.imgzt.setImage(datazt)

        limits = 0., datazt.max()
        bar = pg.ColorBarItem(values=limits, colorMap=self.color_map, limits=limits, rounding=.1,width=15)
        bar.setImageItem(self.imgzt, insert_in=self.p1zt)
  
        # # Custom ROI for selecting an image region
        roi_bounds = QtCore.QRectF(self.t0,self.z0,self.tend-self.t0,self.zend-self.z0)


        # roi = pg.ROI([t0,t0], [tend-t0,-t0+tend],maxBounds=roi_bounds)

        self.roizt = pg.ROI([(self.tend+self.t0)/2-(self.tend-self.t0)/8,(self.zend+self.z0)/2-(-self.z0+self.zend)/8], [(self.tend-self.t0)/4,(-self.z0+self.zend)/4],maxBounds=roi_bounds)
        self.roizt.addScaleHandle([1, 1], [0.5, 0.5])
        self.roizt.addTranslateHandle([0.5, 0.5])
        self.roizt.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        # roi.addScaleHandle([0, 0.5], [0.5, 0.5])

        self.roi2zt = pg.ROI(pos=self.roizt.pos(),size=(0,0),maxBounds=roi_bounds,movable=False,pen=(0,255,0),removable=True)
        self.p1zt.addItem(self.roizt)
        self.roizt.setZValue(10)  # make sure ROI is drawn above image
        self.p1zt.addItem(self.roi2zt)
        self.roi2zt.setZValue(5)  # make sure ROI is drawn above image
        
        self.roizt.sigRegionChanged.connect(self.updatePlotzt)
        self.roizt.sigClicked.connect(self.roi1Clickedzt)
        self.roizt.sigRegionChangeFinished.connect(self.updatePlotFinishedzt)
        
    def roi1Clickedzt(self,roi,event):
        self.imgzt.setLevels((np.min(roi.getArrayRegion(self.imgzt.image,self.imgzt)),np.max(roi.getArrayRegion(self.imgzt.image,self.imgzt))))
    
    def updatePlotzt(self):
        selected = self.roi.getState()
        selectedzt = self.roizt.getState()
        self.p2linezt.setData(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[3]:slice(selectedzt['pos'][0],selectedzt['pos'][0]+selectedzt['size'][0])}].sum(axis=(0,1,3)).values)
        self.p3linezt.setData(self.xr_data.coords[self.xr_data.dims[3]].values,self.xr_data.loc[{self.xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),self.xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),self.xr_data.dims[2]:slice(selectedzt['pos'][1],selectedzt['pos'][1]+selectedzt['size'][1])}].sum(axis=(0,1,2)).values)
    
    def updatePlotFinishedzt(self):
        selectedzt = self.roizt.getState()
        posz=self.zRegionzt.getRegion()
        post=self.tRegionzt.getRegion()
        self.p1zt.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[3]} = {post[0]:.3f} + {post[1] - post[0]:.3f}  <br>"
        f"EDC = {selectedzt['pos'][0] + selectedzt['size'][0] / 2:.3f} <span>&#177;</span> {selectedzt['size'][0] / 2:.3f}  "
        f" | DDC = {selectedzt['pos'][1] + selectedzt['size'][1] / 2:.3f} <span>&#177;</span> {selectedzt['size'][1] / 2:.3f}",
        wrap=True
        )
        
    def init_yz_zcut(self):
        self.p2yz = self.win.addPlot(colspan=1)
        self.p2viewyz=self.p2yz.getViewBox()
        # p2view.setMouseMode(p2view.RectMode)
        self.p2yz.setMaximumHeight(100)
        self.p2lineyz=self.p2yz.plot(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.sum(axis=(0,1,3)).values, clear=True)
        self.p2line2yz=self.p2yz.plot()
        self.zRegionyz = pg.LinearRegionItem([self.z0, self.zend], orientation='vertical', movable=True,bounds=[self.z0, self.zend])
        self.p2yz.addItem(self.zRegionyz)
        self.p2yz.setLabel("bottom", self.xr_data.dims[2])  # Set x-axis label
        self.zRegionyz.sigRegionChangeFinished.connect(self.updateRegionyz)
    
    def updateRegionyz(self):
        posz=self.zRegionyz.getRegion()
        posy=self.yRegionyz.getRegion()
        selectedyz = self.roiyz.getState()
        # p1yz.setTitle(xr_data.dims[2]+f" ({posz[0]:.3f}, {posz[1]-posz[0]:.3f}) "+xr_data.dims[1]+f" ({posx[0]:.3f}, {posx[1]-posx[0]:.3f}) "+"<br>EDC"+f" ({selectedyz['pos'][0]+selectedyz['size'][0]/2:.3f}, {selectedyz['size'][0]/2:.3f}) MDC"+f" ({selectedyz['pos'][1]+selectedyz['size'][1]/2:.3f}, {selectedyz['size'][1]/2:.3f})",wrap=True)
        self.p1yz.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[1]} = {posy[0]:.3f} + {posy[1] - posy[0]:.3f}  <br>"
        f"EDC = {selectedyz['pos'][0] + selectedyz['size'][0] / 2:.3f} <span>&#177;</span> {selectedyz['size'][0] / 2:.3f}  "
        f" | MDC = {selectedyz['pos'][1] + selectedyz['size'][1] / 2:.3f} <span>&#177;</span> {selectedyz['size'][1] / 2:.3f}",
        wrap=True
        )
        
        
    def init_zt_zcut(self):
        self.p2zt = self.win.addPlot(colspan=1)
        self.p2viewzt=self.p2zt.getViewBox()
        # p2view.setMouseMode(p2view.RectMode)
        self.p2zt.setMaximumHeight(100)
        self.p2linezt=self.p2zt.plot(self.xr_data.coords[self.xr_data.dims[2]].values,self.xr_data.sum(axis=(0,1,3)).values, clear=True)
        self.p2line2zt=self.p2zt.plot()
        self.zRegionzt = pg.LinearRegionItem([self.z0, self.zend], orientation='vertical', movable=True,bounds=[self.z0, self.zend])
        self.p2zt.addItem(self.zRegionzt)
        self.p2zt.setLabel("bottom", self.xr_data.dims[2])  # Set x-axis label
        self.zRegionzt.sigRegionChangeFinished.connect(self.updateRegionzt)
    
    def updateRegionzt(self):
        posz=self.zRegionzt.getRegion()
        post=self.tRegionzt.getRegion()
        selectedzt = self.roizt.getState()
        # p1zt.setTitle(xr_data.dims[2]+f" ({posz[0]:.3f}, {posz[1]-posz[0]:.3f}) "+xr_data.dims[1]+f" ({posx[0]:.3f}, {posx[1]-posx[0]:.3f}) "+"<br>EDC"+f" ({selectedzt['pos'][0]+selectedzt['size'][0]/2:.3f}, {selectedzt['size'][0]/2:.3f}) MDC"+f" ({selectedzt['pos'][1]+selectedzt['size'][1]/2:.3f}, {selectedzt['size'][1]/2:.3f})",wrap=True)
        self.p1zt.setTitle(
        f"{self.xr_data.dims[2]} = {posz[0]:.3f} + {posz[1] - posz[0]:.3f}  "
        f" | {self.xr_data.dims[3]} = {post[0]:.3f} + {post[1] - post[0]:.3f}  <br>"
        f"EDC = {selectedzt['pos'][0] + selectedzt['size'][0] / 2:.3f} <span>&#177;</span> {selectedzt['size'][0] / 2:.3f}  "
        f" | DDC = {selectedzt['pos'][1] + selectedzt['size'][1] / 2:.3f} <span>&#177;</span> {selectedzt['size'][1] / 2:.3f}",
        wrap=True
        )
        
        
    def init_yz_ycut(self):
        self.p3yz = self.win.addPlot(colspan=1)
        self.p3yz.setMaximumHeight(100)
        self.p3viewyz=self.p3yz.getViewBox()
        self.p3lineyz=self.p3yz.plot(self.xr_data.coords[self.xr_data.dims[1]].values,self.xr_data.sum(axis=(0,2,3)).values, clear=True)
        self.p3lineyz2=self.p3yz.plot()
        self.yRegionyz = pg.LinearRegionItem([self.y0, self.yend], orientation='vertical', movable=True,bounds=[self.y0, self.yend])
        self.p3yz.addItem(self.yRegionyz)
        self.p3yz.setLabel("bottom", self.xr_data.dims[1])  # Set x-axis label
        self.yRegionyz.sigRegionChangeFinished.connect(self.updateRegionyz)
        
    def init_zt_tcut(self):
        self.p3zt = self.win.addPlot(colspan=1)
        self.p3zt.setMaximumHeight(100)
        self.p3viewzt=self.p3zt.getViewBox()
        self.p3linezt=self.p3zt.plot(self.xr_data.coords[self.xr_data.dims[3]].values,self.xr_data.sum(axis=(0,1,2)).values, clear=True)
        self.p3linezt2=self.p3zt.plot()
        self.tRegionzt = pg.LinearRegionItem([self.t0, self.tend], orientation='vertical', movable=True,bounds=[self.t0, self.tend])
        self.p3zt.addItem(self.tRegionzt)
        self.p3zt.setLabel("bottom", self.xr_data.dims[3])  # Set x-axis label
        self.tRegionzt.sigRegionChangeFinished.connect(self.updateRegionzt)


    def keypress(self,event):
        # print(event.key())
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            pos=self.tRegion.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.ts)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.ts)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.ts,pos[1]-10*self.ts)
            else:
                pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.tRegion.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            pos=self.tRegion.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.ts)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.ts)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.ts,pos[1]+10*self.ts)
            else:
                pos_new=(pos[0]+self.ts,pos[1]+self.ts)
            self.tRegion.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            pos=self.zRegion.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.zs,pos[1]-10*self.zs)
            else:
                pos_new=(pos[0]-self.zs,pos[1]-self.zs)
            self.zRegion.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Up:
            # print("right")
            pos=self.zRegion.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.zs,pos[1]+10*self.zs)
            else:
                pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            # pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.zRegion.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.img.setLevels(self.img.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2.listDataItems()[0]
            tb1_line2=self.p2.listDataItems()[1]
            tb2_line1=self.p3.listDataItems()[0]
            tb2_line2=self.p3.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2.setPos(self.roi.pos())
                self.roi2.setSize(self.roi.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2.setSize((0,0))
                self.win.request_draw()

    def keypressxz(self,event):
        # print(event.key())
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            pos=self.zRegionxz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.zs,pos[1]-10*self.zs)
            else:
                pos_new=(pos[0]-self.zs,pos[1]-self.zs)
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.zRegionxz.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Up:
            # print("right")
            pos=self.zRegionxz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.zs,pos[1]+10*self.zs)
            else:
                pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.zRegionxz.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            pos=self.xRegionxz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.xs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.xs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.xs,pos[1]-10*self.xs)
            else:
                pos_new=(pos[0]-self.xs,pos[1]-self.xs)
            self.xRegionxz.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            pos=self.xRegionxz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.xs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.xs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.xs,pos[1]+10*self.xs)
            else:
                pos_new=(pos[0]+self.xs,pos[1]+self.xs)
            # pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.xRegionxz.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.imgxz.setLevels(self.imgxz.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2xz.listDataItems()[0]
            tb1_line2=self.p2xz.listDataItems()[1]
            tb2_line1=self.p3xz.listDataItems()[0]
            tb2_line2=self.p3xz.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2xz.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2xz.setPos(self.roixz.pos())
                self.roi2xz.setSize(self.roixz.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2xz.setSize((0,0))
                self.win.request_draw()       
            # print(tb1)
            # tb1.setPen((0,255,0))
            # p2.plot(tb1x,tb1y,clear=True,pen=(255,255))
            # self.roi.setZValue(10)
            # print(p2.listDataItems())
            # tb2=self.p3.listDataItems().copy()
            # tb2.setPen((0,255,0))
            # self.p3.addItem(tb2)


        # setRegion(self, rgn):


    def keypressyz(self,event):
        # print(event.key())
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            pos=self.zRegionyz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.zs,pos[1]-10*self.zs)
            else:
                pos_new=(pos[0]-self.zs,pos[1]-self.zs)
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.zRegionyz.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Up:
            # print("right")
            pos=self.zRegionyz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.zs,pos[1]+10*self.zs)
            else:
                pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.zRegionyz.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            pos=self.yRegionyz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.ys,pos[1]-10*self.ys)
            else:
                pos_new=(pos[0]-self.ys,pos[1]-self.ys)
            self.yRegionyz.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            pos=self.yRegionyz.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.ys,pos[1]+10*self.ys)
            else:
                pos_new=(pos[0]+self.ys,pos[1]+self.ys)
            # pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.yRegionyz.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.imgyz.setLevels(self.imgyz.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2yz.listDataItems()[0]
            tb1_line2=self.p2yz.listDataItems()[1]
            tb2_line1=self.p3yz.listDataItems()[0]
            tb2_line2=self.p3yz.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2yz.setPos(self.roiyz.pos())
                self.roi2yz.setSize(self.roiyz.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2yz.setSize((0,0))
                self.win.request_draw()       



    def keypresszt(self,event):
        # print(event.key())
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            pos=self.zRegionzt.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.zs,pos[1]-10*self.zs)
            else:
                pos_new=(pos[0]-self.zs,pos[1]-self.zs)
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.zRegionzt.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Up:
            # print("right")
            pos=self.zRegionzt.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.zs)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.zs)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.zs,pos[1]+10*self.zs)
            else:
                pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.zRegionzt.setRegion(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            pos=self.tRegionzt.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]-self.ts)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]-10*self.ts)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]-10*self.ts,pos[1]-10*self.ts)
            else:
                pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.tRegionzt.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            pos=self.tRegionzt.getRegion()
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                pos_new=(pos[0],pos[1]+self.ts)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                pos_new=(pos[0],pos[1]+10*self.ts)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(pos[0]+10*self.ts,pos[1]+10*self.ts)
            else:
                pos_new=(pos[0]+self.ts,pos[1]+self.ts)
            # pos_new=(pos[0]+self.zs,pos[1]+self.zs)
            self.tRegionzt.setRegion(pos_new)
            # updateself.tRegion()
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.imgyz.setLevels(self.imgyz.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2zt.listDataItems()[0]
            tb1_line2=self.p2zt.listDataItems()[1]
            tb2_line1=self.p3zt.listDataItems()[0]
            tb2_line2=self.p3zt.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2zt.setPos(self.roizt.pos())
                self.roi2zt.seself.tsize(self.roizt.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2zt.setSize((0,0))
                self.win.request_draw()





    def keypressimg(self,event):
        # print(event.key())
        selected = self.roi.getState()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]-10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:

                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roi.setSize(size_new)
            self.roi.setPos(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]+10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roi.setPos(pos_new)
            self.roi.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]-10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roi.setPos(pos_new)
            self.roi.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Up:
             # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]+10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roi.setPos(pos_new)
            self.roi.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.img.setLevels(self.img.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2.listDataItems()[0]
            tb1_line2=self.p2.listDataItems()[1]
            tb2_line1=self.p3.listDataItems()[0]
            tb2_line2=self.p3.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2.setPos(self.roi.pos())
                self.roi2.setSize(self.roi.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2.setSize((0,0))
                self.win.request_draw()


    def keypressimgxz(self,event):
        # print(event.key())
        selected = self.roixz.getState()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]-10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:

                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roixz.setSize(size_new)
            self.roixz.setPos(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]+10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roixz.setPos(pos_new)
            self.roixz.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]-10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roixz.setPos(pos_new)
            self.roixz.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Up:
             # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]+10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roixz.setPos(pos_new)
            self.roixz.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.imgxz.setLevels(self.imgxz.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2xz.listDataItems()[0]
            tb1_line2=self.p2xz.listDataItems()[1]
            tb2_line1=self.p3xz.listDataItems()[0]
            tb2_line2=self.p3xz.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2xz.setPos(self.roixz.pos())
                self.roi2xz.setSize(self.roixz.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2xz.setSize((0,0))
                self.win.request_draw()



    def keypressimgyz(self,event):
        # print(event.key())
        selected = self.roiyz.getState()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]-10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:

                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roiyz.setSize(size_new)
            self.roiyz.setPos(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]+10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roiyz.setPos(pos_new)
            self.roiyz.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]-10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roiyz.setPos(pos_new)
            self.roiyz.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Up:
             # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]+10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roiyz.setPos(pos_new)
            self.roiyz.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.imgyz.setLevels(self.imgyz.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2yz.listDataItems()[0]
            tb1_line2=self.p2yz.listDataItems()[1]
            tb2_line1=self.p3yz.listDataItems()[0]
            tb2_line2=self.p3yz.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2yz.setPos(self.roiyz.pos())
                self.roi2yz.setSize(self.roiyz.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2yz.setSize((0,0))
                self.win.request_draw()


    def keypressimgzt(self,event):
        # print(event.key())
        selected = self.roizt.getState()
        if event.key() == QtCore.Qt.Key_Left:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]-10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:

                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roizt.setSize(size_new)
            self.roizt.setPos(pos_new)
            self.win.request_draw()
            # updateself.tRegion()
            # print(pos)
        if event.key() == QtCore.Qt.Key_Right:
            # print("right")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1])
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1])
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0]+10*self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1])
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roizt.setPos(pos_new)
            self.roizt.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Down:
            # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]-2*self.xs,selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0]+self.xs,selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]-2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]-10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roizt.setPos(pos_new)
            self.roizt.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_Up:
             # print("left")
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                size_new=(selected["size"][0]+2*self.xs,selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0]-self.xs,selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                size_new=(selected["size"][0],selected["size"][1]+2*self.ys)
                pos_new=(selected["pos"][0],selected["pos"][1]-self.ys)
            elif event.modifiers() == QtCore.Qt.AltModifier:
                pos_new=(selected["pos"][0],selected["pos"][1]+10*self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            else:
                pos_new=(selected["pos"][0],selected["pos"][1]+self.ys)
                size_new=(selected["size"][0],selected["size"][1])
            # pos_new=(pos[0]-self.ts,pos[1]-self.ts)
            self.roizt.setPos(pos_new)
            self.roizt.setSize(size_new)
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_A:
            self.imgzt.setLevels(self.imgzt.quickMinMax())
            self.win.request_draw()
        if event.key() == QtCore.Qt.Key_B:
            # print("here")
            # print(p2.listDataItems())
            # p2.plot(xr_data.coords[xr_data.dims[2]].values,xr_data.loc[{xr_data.dims[0]:slice(selected['pos'][0],selected['pos'][0]+selected['size'][0]),xr_data.dims[1]:slice(selected['pos'][1],selected['pos'][1]+selected['size'][1]),xr_data.dims[3]:slice(*self.tRegion.getRegion())}].sum(axis=(0,1,3)).values, clear=False, pen=(0,255,0))
            tb1_line1=self.p2zt.listDataItems()[0]
            tb1_line2=self.p2zt.listDataItems()[1]
            tb2_line1=self.p3zt.listDataItems()[0]
            tb2_line2=self.p3zt.listDataItems()[1]
            (tb1x,tb1y)=tb1_line1.getData()
            (tb1xx,tb1yy)=tb1_line2.getData()
            if tb1xx is None:
                tb1_line2.setData(tb1x,tb1y,pen=(0,255,0))


                (tb2x,tb2y)=tb2_line1.getData()
                tb2_line2.setData(tb2x,tb2y,pen=(0,255,0))

                # selected = self.roi.geself.tstate()

                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=self.roi.size(),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                # self.roi2 = pg.self.roi(pos=self.roi.pos(),size=(0,0),maxBounds=self.roi_bounds,movable=False,pen=(0,255,0),removable=True)
                self.roi2zt.setPos(self.roizt.pos())
                self.roi2zt.setSize(self.roizt.size())
                self.win.request_draw()
                # self.roi2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
                # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
                # p1.addItem(self.roi2)
                # self.roi2.setZValue(5)  # make sure self.roi is drawn above image
                # self.roi2.sigClicked.connect(self.roi2Clicked)
            else:
                # print('not none')
                tb1_line2.setData()
                tb2_line2.setData()
                self.roi2zt.setSize((0,0))
                self.win.request_draw()
				
				
				
def loadh5(dataset_key ,file_path):				
    with h5py.File(file_path, 'r') as h5file:
    # Assume the dataset is stored under a specific key in the HDF5 file
        data = h5file[dataset_key][:]
        size=h5file[dataset_key].shape
        dims=[]
        coords={}
        i=0
        for dim in size:
            dims.append(h5file[dataset_key].attrs["axis"+str(i)])
            coords[dims[i]]=np.linspace(start=h5file[dataset_key].attrs["axis"+str(i)+"_start"],stop=h5file[dataset_key].attrs["axis"+str(i)+"_stop"], num=dim)
            i+=1

		# Optionally, retrieve coordinate information if available
		# Replace 'x', 'y', 'z', 't' with the actual dimension names if availab
		# Create an xarray DataArray from the loaded data
    xr_data = xr.DataArray(data,dims=dims,coords=coords)
    return(xr_data)
