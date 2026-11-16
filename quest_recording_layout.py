"""Collection layout; lifecycle and commands remain in RecordingPanel."""
from pathlib import Path
from gi.repository import Gtk, GLib
from quest_dashboard_layout import vertical, pack, field_grid


def build_recording_page(self, owner):
    self.banner = Gtk.Label(label="数据采集：未准备",xalign=0)
    self.banner.set_line_wrap(True)
    self.banner.set_max_width_chars(90)
    owner.header_status_box.pack_start(self.banner,False,False,0)
    self.notebook=owner.workspace_notebook
    page=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=12,margin=12)
    page_scroll=Gtk.ScrolledWindow()
    page_scroll.set_policy(Gtk.PolicyType.NEVER,Gtk.PolicyType.AUTOMATIC)
    page_scroll.add(page)
    self.notebook.append_page(page_scroll,Gtk.Label(label="数据采集"))


    self.dataset = Gtk.Entry(text="xarm7_demonstrations")
    self.folder = Gtk.Entry(text=str(Path(self.config["dataset_root"])/"raw"))
    self.task = Gtk.Entry()
    self.task.set_placeholder_text("例如：Pick up the cup and place it in the tray.")
    self.task.set_tooltip_text("任务输入框中空格用于输入；Esc 和顶部停止按钮仍可停止遥操。")
    settings=Gtk.Expander(label='数据集与保存位置（准备采集前设置）')
    settings.add(field_grid([('数据集名称',self.dataset),('原始记录父目录',self.folder)]))
    page.pack_start(settings,False,False,0)
    page.pack_start(field_grid([('本条任务',self.task)]),False,False,0)
    self.controls=Gtk.Box(spacing=8)
    self.prepare=self.button("准备采集 / 相机预览",self.start_process)
    self.start=self.button("开启本条采集",lambda *_:self.send("start",task=self.task.get_text()))
    self.save=self.button("结束本条 / 判定并保存",self.save_episode)
    self.discard=self.button("判定并丢弃",lambda *_:self.save_episode(keep=False))
    self.stop=self.button("关闭采集器",self.shutdown)
    devices=Gtk.Box(spacing=8)
    pack(devices,self.prepare,self.stop)
    page.pack_start(devices,False,False,0)
    page.pack_start(self.controls,False,False,0)
    self.start.get_style_context().add_class("suggested-action")
    self.detail=Gtk.Label(label="录制独立运行：开启本条后只记录有效遥操；结束、判定、关闭录制不会停止遥操。",xalign=0)
    self.detail.set_line_wrap(True)
    self.detail.set_max_width_chars(90)
    self.detail.set_selectable(True)
    page.pack_start(self.detail,False,False,0)
    preview=Gtk.Box(spacing=12)
    self.images={}
    for camera in self.config["cameras"]:
        frame=Gtk.Frame(label=camera["name"])
        image=Gtk.Image()
        image.set_size_request(320,240)
        frame.add(image)
        preview.pack_start(frame,True,True,0)
        self.images[camera["name"]]=image
    page.pack_start(preview,False,False,0)
    home_box=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=6)
    home_row=Gtk.Box(spacing=8)
    self.home_buttons={}
    for command,label in (("release","禁用遥操（复位前）"),("home_save","记录 / 更新初始位姿"),
                          ("home_check","检查复位路径"),("home_start","返回初始位姿")):
        button=Gtk.Button(label=label)
        button.connect('clicked',lambda _,c=command:self.home_action(c))
        home_row.pack_start(button,False,False,0)
        self.home_buttons[command]=button
    home_box.pack_start(home_row,False,False,0)
    speed_row=Gtk.Box(spacing=8)
    speed_row.pack_start(Gtk.Label(label='复位速度 °/s'),False,False,0)
    self.home_speed=Gtk.SpinButton(adjustment=owner.home_speed_setting.get_adjustment(),digits=0)
    self.home_speed.set_tooltip_text(owner.home_speed_setting.get_tooltip_text())
    speed_row.pack_start(self.home_speed,False,False,0)
    home_box.pack_start(speed_row,False,False,0)
    self.home_status=Gtk.Label(xalign=0)
    self.home_status.set_line_wrap(True)
    self.home_status.set_max_width_chars(90)
    home_box.pack_start(self.home_status,False,False,0)
    home=Gtk.Expander(label='轮次间复位 / 初始位姿')
    home.add(home_box)
    page.pack_start(home,False,False,0)
    self.export_box=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=6)
    export_row=Gtk.Box(spacing=10)
    self.format=Gtk.ComboBoxText()
    self.format.append("smooth269","269平滑同流程 · 30Hz / 0.5mm / 自动校验")
    self.format.append("lerobot21","LeRobot v2.1 · OpenPI / ACT / 其他模型")
    self.format.append("robotwin","RoboTwin / XPolicyLab · HDF5")
    self.format.set_active_id("smooth269")
    export_row.pack_start(self.format,True,True,0)
    export_row.pack_start(Gtk.Label(label='FPS'),False,False,0)
    self.export_fps=Gtk.SpinButton.new_with_range(1,60,1)
    self.export_fps.set_value(30)
    self.export_fps.set_tooltip_text('269平滑模式固定30Hz；其他格式可自行设置。')
    export_row.pack_start(self.export_fps,False,False,0)
    self.scan=Gtk.Button(label='检查数据')
    self.scan.connect('clicked',lambda *_:self.start_export(scan_only=True))
    export_row.pack_start(self.scan,False,False,0)
    self.export=Gtk.Button(label="导出")
    self.export.connect("clicked",self.start_export)
    export_row.pack_start(self.export,False,False,0)
    self.export_box.pack_start(export_row,False,False,0)
    self.clean_export=Gtk.CheckButton(label='清洗导出：跳过不合格整条轨迹，保留原始数据')
    self.clean_export.set_active(True)

    self.report_button=Gtk.Button(label='查看检查报告')
    self.report_button.set_sensitive(False)
    self.report_button.connect('clicked',self.show_quality_report)
    export_row.pack_start(self.report_button,False,False,0)
    self.include_failures=Gtk.CheckButton(label="同时导出已保留的失败轨迹（默认仅成功示范）")
    options=Gtk.Expander(label='导出选项 · 默认清洗、仅成功轨迹')
    options.add(pack(vertical(),self.clean_export,self.include_failures))
    self.export_box.pack_start(options,False,False,0)
    def sync_profile(*_):
        fixed=self.format.get_active_id()=='smooth269'
        self.export_fps.set_sensitive(not fixed);self.clean_export.set_sensitive(not fixed);self.include_failures.set_sensitive(not fixed)
        if fixed:self.export_fps.set_value(30);self.clean_export.set_active(True);self.include_failures.set_active(False)
    self.format.connect('changed',sync_profile);sync_profile()
    self.export_status=Gtk.Label(label="检查和导出作用于上方所选整个数据集；列表筛选仅用于浏览。",xalign=0)
    self.export_status.set_line_wrap(True)
    self.export_status.set_max_width_chars(90)
    self.export_status.set_selectable(True)
    self.export_box.pack_start(self.export_status,False,False,0)
    note=Gtk.Label(label="仅在扳机按住且机械臂跟随时记录：机器人/手柄 + 双相机 PNG + 时间戳。\n"
        "检查、清洗和导出请到“数据管理”页。\n"
        "松开暂停写入，再按继续同一条；一次保存、一次导出均为一个 episode。\n"
        "任务完成后，保持按住扳机观察几秒，再松开并结束；成功由人工判定。\n"
        "判定期间录制冻结，选择成功/失败及是否保留；丢弃记录不导出，原始文件仍留存。\n"
        "初始位姿长期保存在 relative_home.json；复位独立控制，录制中复位会把本条标为未完成。\n"
        "OpenPI 需配套 xArm7 数据适配器；不直接套用 ALOHA 双臂配置。",xalign=0)
    note.set_line_wrap(True)
    help_panel=Gtk.Expander(label='采集规则 / 格式说明')
    help_panel.add(note)
    page.pack_start(help_panel,False,False,0)
    self.update_buttons()
    from quest_library_ui import LibraryPanel
    self.library = LibraryPanel(self)
    self.notebook.append_page(owner.device_page,Gtk.Label(label="设备诊断"))
    self.timer=GLib.timeout_add(500,self.refresh)

