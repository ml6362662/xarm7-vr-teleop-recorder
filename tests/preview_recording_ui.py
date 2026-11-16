"""Offline UI screenshot; no receiver, camera, ADB or robot connection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quest_pose_viewer import PoseWindow,Gtk,GLib,Gdk

window=PoseWindow(demo=True)
window.show_all()
window.recording_panel.notebook.set_current_page(1)
def capture():
    pixbuf=Gdk.pixbuf_get_from_window(window.get_window(),0,0,window.get_allocated_width(),window.get_allocated_height())
    pixbuf.savev(str(Path(__file__).resolve().parents[1]/".state/recording_ui_preview.png"),"png",[],[])
    window.destroy()
    return False
GLib.timeout_add(1500,capture)
Gtk.main()
