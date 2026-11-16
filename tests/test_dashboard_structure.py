"""Widget ownership/signal contracts with a small GTK tree double, not a render test."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import quest_dashboard_layout as layout
import quest_recording_layout as collection
import quest_recording_ui as recorder
import quest_library_ui as library


class Widget:
    """Model container ownership and signals; unknown calls fail explicitly."""
    def __init__(self, **kwargs):
        self.props=dict(sensitive=True,expanded=False,**kwargs)
        self.children=[];self.parent=None;self.signals=defaultdict(list)
        self.style=SimpleNamespace(add_class=lambda *_:None)
    def add(self,child):
        if child.parent is not None:
            raise AssertionError('widget already has a parent')
        self.children.append(child);child.parent=self
    def remove(self,child): self.children.remove(child);child.parent=None
    def pack_start(self,child,*_): self.add(child)
    def pack1(self,child,*_): self.add(child)
    def pack2(self,child,*_): self.add(child)
    def attach(self,child,*_): self.add(child)
    def append_page(self,child,label): self.add(child);child.tab=label.props['label']
    def get_text(self): return self.props.get('text',self.props.get('label',''))
    def get_parent(self): return self.parent
    def get_children(self): return self.children
    def get_style_context(self): return self.style
    def get_selection(self): return self
    def get_adjustment(self): return self
    def append_column(self,*_): pass
    def insert_column(self,*_): pass
    def __iter__(self): return iter(())
    def connect(self,signal,handler,*args): self.signals[signal].append((handler,args))
    def append(self,*_): pass
    def clear(self): self.props['cleared']=True
    def emit(self,signal):
        for handler,args in self.signals[signal]: handler(self,*args)
    def __getattr__(self,name):
        if name in ('set_policy','set_size_request','set_range','set_property'):
            return lambda *args:self.props.update({name:args})
        if name.startswith('set_'):
            return lambda value:self.props.update({name[4:]:value})
        if name.startswith('get_'):
            return lambda:self.props.get(name[4:])
        raise AttributeError(name)
    @classmethod
    def new_with_range(cls,*_): return cls()


def fake_gtk():
    types=('Entry Box Label Grid Frame ScrolledWindow Button SpinButton CheckButton ComboBoxText '
           'DrawingArea Notebook Expander Paned Image Scale TreeView CellRendererText')
    result={key:type(key,(Widget,),{}) for key in types.split()}
    result.update(Orientation=SimpleNamespace(VERTICAL=1,HORIZONTAL=0),
        PolicyType=SimpleNamespace(AUTOMATIC=1,NEVER=0),
        CssProvider=lambda:SimpleNamespace(load_from_data=lambda *_:None),
        StyleContext=SimpleNamespace(add_provider_for_screen=lambda *_:None),
        STYLE_PROVIDER_PRIORITY_APPLICATION=600,
        ListStore=lambda *_:Widget(),TreeViewColumn=lambda *_args,**_kwargs:Widget())
    return SimpleNamespace(**result)


class DashboardStructureTests(unittest.TestCase):
    def setUp(self):
        gtk=fake_gtk()
        self.patches=[patch.object(module,'Gtk',gtk) for module in (layout,collection,recorder,library)]
        self.patches += [patch.object(layout,'Gdk',SimpleNamespace(Screen=SimpleNamespace(get_default=lambda:None))),
                         patch.object(collection,'GLib',SimpleNamespace(timeout_add=lambda *_:1))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.owner=Widget();self.owner.lan_host=None;self.owner.demo=False;self.owner.closing=False
        methods=('connect_quest stop zero export connect_robot robot_command quest_app_action poll_quest_status '
                 'draw_plot matrix_choice_changed confirm_matrix_choice apply_settings apply_responsive_settings '
                 'home_speed_changed recovery_action settings_edited')
        for method in methods.split():setattr(self.owner,method,Mock())
        layout.build_dashboard(self.owner)
        self.panel=recorder.RecordingPanel(self.owner,self.owner.workspace_root)
        self.addCleanup(self.panel.library.pool.shutdown,wait=False)

    def descendants(self,widget):
        result={widget}
        for child in widget.children:result.update(self.descendants(child))
        return result

    def test_four_pages_and_stop_outside_notebook(self):
        owner=self.owner
        self.assertEqual([p.tab for p in owner.workspace_notebook.children],
                         ['遥操控制','数据采集','数据管理','设备诊断'])
        self.assertNotIn(owner.robot_stop,self.descendants(owner.workspace_notebook))
        self.assertIn(owner.robot_stop,self.descendants(owner))
        self.assertIn(self.panel.banner,self.descendants(owner.header_status_box))

    def test_every_exposed_widget_is_reachable_without_duplicate_parent(self):
        reachable=self.descendants(self.owner)
        for obj in (self.owner,self.panel,self.panel.library):
            for name,value in vars(obj).items():
                if isinstance(value,Widget) and name not in ('store',):
                    with self.subTest(name=name):self.assertIn(value,reachable)
        self.assertEqual(len(self.owner.values),3)
        for widgets in (self.owner.values,self.owner.deltas,self.owner.rotation_axis_settings):
            for widget in widgets:self.assertIn(widget,reachable)

    def test_existing_motion_actions_keep_their_single_original_handler(self):
        for name,command in [('robot_stop','stop'),('robot_release','release'),
                             ('robot_enable','allow_controller'),('gripper_close_button','gripper_close')]:
            button=getattr(self.owner,name)
            self.assertEqual(len(button.signals['clicked']),1)
            button.emit('clicked')
            self.owner.robot_command.assert_called_with(command)
        self.owner.recovery_action.assert_not_called()

    def test_failed_dataset_switch_cannot_export_previous_dataset(self):
        lib=self.panel.library
        lib.root=Path('/tmp/old');lib.parent=Path('/tmp')
        lib.datasets.set_active_id('new')
        with patch.object(lib,'submit') as submit:
            lib.dataset_changed()
            self.assertIsNone(lib.root)
            self.assertFalse(self.panel.export.get_sensitive())
            apply=submit.call_args.args[1]
            apply(None,'读取失败')
        self.assertIsNone(lib.root)
        self.assertFalse(self.panel.scan.get_sensitive())
        self.assertIn('读取失败',lib.notice.get_text())

    def test_tuning_does_not_apply_automatically_and_home_speed_is_separate(self):
        self.owner.speed_setting.emit('value-changed')
        self.owner.settings_edited.assert_called_once()
        self.owner.apply_settings.assert_not_called()
        self.owner.settings_edited.reset_mock()
        self.owner.home_speed_setting.emit('value-changed')
        self.owner.home_speed_changed.assert_called_once()
        self.owner.settings_edited.assert_not_called()
        for expander in (self.owner.advanced_panel,self.owner.recovery_panel,self.owner.diagnostics_panel):
            self.assertFalse(expander.get_expanded())


if __name__=='__main__':unittest.main()
