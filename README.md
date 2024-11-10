Widget utilizes jupyter remote-frame-buffer (jupyter_rfb) to run pyqtgraph inside of jupyter notebook cell. This enables to run gui directly on a jupyter server. Usage:

from view4D import View4D
view=View4D(xarray_data_4D)

to take a snaphot of widget state use view.win.snapshot()

keyboard actions:
a- autoscale color map range

b- toggle a reference box

arrow keys- move ROIs or line selections depending on the focus. Change focus by clicking on an image or a plot

Control/Shift/Alt+arrow keys- resize/faster move/ etc.

![Screenshot_view4D](https://github.com/user-attachments/assets/b5424b06-8901-442e-9d84-09a6abafaffb)
