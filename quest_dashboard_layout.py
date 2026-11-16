"""Explicit dashboard hierarchy. Widgets keep the original command handlers."""
import math
import gi
gi.require_version('Gtk','3.0')
gi.require_version('Gdk','3.0')
from gi.repository import Gtk, Gdk
from quest_control_limits import TUNING_LIMITS, HOME_SPEED_DEFAULT, HOME_SPEED_LIMITS


def vertical(spacing=8, margin=0):
    return Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=spacing, margin=margin)


def pack(box, *widgets):
    for widget in widgets:
        parent = widget.get_parent()
        if parent is not None:
            parent.remove(widget)
        box.pack_start(widget, False, False, 0)
    return box


def field_grid(pairs):
    grid = Gtk.Grid(column_spacing=12, row_spacing=8)
    for row, (text, widget) in enumerate(pairs):
        parent = widget.get_parent()
        if parent is not None:
            parent.remove(widget)
        label = Gtk.Label(label=text, xalign=0)
        widget.set_hexpand(True)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(widget, 1, row, 1, 1)
    return grid


def section(title, content):
    frame = Gtk.Frame(label=title)
    frame.add(content)
    return frame


def scroll_page(content):
    scroll = Gtk.ScrolledWindow()
    scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
    scroll.add(content)
    return scroll


def create_controls(self):
    """Create each control once; handlers remain methods of the owning window."""
    def button(name, text, callback):
        widget = Gtk.Button(label=text)
        widget.connect('clicked', callback)
        setattr(self, name, widget)
        return widget

    def label(name, text=''):
        widget = Gtk.Label(label=text, xalign=0)
        widget.set_line_wrap(True)
        widget.set_max_width_chars(80)
        widget.set_selectable(True)
        setattr(self, name, widget)
        return widget

    def spin(name, bounds, step, value, digits=0):
        widget = Gtk.SpinButton.new_with_range(*bounds, step)
        widget.set_digits(digits)
        widget.set_value(value)
        setattr(self, name, widget)

    button('connect_button', '连接手柄', self.connect_quest)
    button('receiver_stop', '停止接收', lambda *_: self.stop())
    button('diagnostic_zero', '设为诊断零点', self.zero)
    button('diagnostic_export', '导出位置 CSV', self.export)
    button('robot_connect', '连接机械臂', self.connect_robot)
    for name, text, command in (
        ('robot_enable','启用遥操','allow_controller'), ('robot_takeover','接管 / 重设起点','takeover'),
        ('robot_release','禁用遥操','release'), ('robot_stop','停止 [空格]','stop'),
        ('gripper_open_button','打开夹爪','gripper_open'), ('gripper_close_button','闭合夹爪','gripper_close')):
        button(name, text, lambda _button, cmd=command: self.robot_command(cmd))
    for name, text, action in (('quest_open','打开头显输入','start'),
                              ('quest_restart','重启头显输入','restart'), ('quest_wake','唤醒头显','wake')):
        button(name, text, lambda _button, a=action: self.quest_app_action(a))
        getattr(self,name).set_tooltip_text('先确认机械臂释放控制，再操作头显；不会自动恢复遥操')
    self.gripper_open_button.set_sensitive(False)
    self.gripper_close_button.set_sensitive(False)
    button('quest_inspect','检查设备状态',lambda *_: self.poll_quest_status())
    for name, text in {
        'status':'手柄输入：未连接', 'detail':'', 'quest_app_status':'头显：等待检查',
        'quest_app_note':'打开头显输入，在头显页面进入 Pass-through。',
        'button_status':'扳机 / B / A：等待真实按键数据', 'gripper_status':'夹爪：未连接',
        'robot_rotation':'实际 / 目标旋转：—', 'robot_status':'机械臂：未连接 · 连接后只读预览',
        'robot_tcp':'实际 / 目标 TCP：—', 'robot_stats':'按住扳机跟随，松开暂停',
        'robot_error':'', 'mapping':'方向映射：等待读取', 'speed_feedback':'实测速率 / 跟随误差：—',
        'range_status':'移动范围由机器人可达性及关节限位约束',
        'orientation':'姿态 / 追踪更新率：—', 'counters':'位姿包：0', 'movement':'位置变化范围：—',
        'instructions':'先只读核对方向和按键，再启用遥操。扳机按住跟随、松开暂停；B 单击切换夹爪。',
    }.items():
        label(name, text)
    self.values = [Gtk.Label(label='—', xalign=0) for _ in range(3)]
    self.deltas = [Gtk.Label(label='相对诊断零点：—', xalign=0) for _ in range(3)]
    self.plot = Gtk.DrawingArea()
    self.plot.set_size_request(640, 220)
    self.plot.connect('draw', self.draw_plot)
    definitions = (
        ('speed_setting','speed_mm_s',5,50,0), ('scale_setting','translation_scale',.05,1,2),
        ('acceleration_setting','acceleration_mm_s2',50,300,0),
        ('tracking_lead_setting','tracking_lead_mm',1,5,0),
        ('rotation_scale_setting','rotation_scale',.05,1,2),
        ('rotation_speed_setting','rotation_speed_deg_s',5,30,0),
        ('rotation_acceleration_setting','rotation_acceleration_deg_s2',10,90,0),
    )
    for name,key,step,value,digits in definitions:
        spin(name,TUNING_LIMITS[key],step,value,digits)
    spin('joint_speed_setting',tuple(math.degrees(v) for v in TUNING_LIMITS['joint_speed_rad_s']),1,math.degrees(.5),1)
    for name,bounds,step,value in (
        ('gripper_speed_setting',(15,225),5,50), ('gripper_force_setting',(1,100),1,30),
        ('gripper_open_setting',(0,84),1,84), ('gripper_close_setting',(0,84),1,0)):
        spin(name,bounds,step,value)
    self.tracking_lead_setting.set_tooltip_text('下发目标领先实测 TCP 的距离上限，实际还受速度与关节约束。不是总移动范围。')
    self.rotation_enabled_setting=Gtk.CheckButton(label='启用相对旋转')
    self.rotation_enabled_setting.set_active(True)
    self.rotation_axis_settings=[]
    for name in ('Rx','Ry','Rz'):
        field=Gtk.CheckButton(label=name)
        field.set_active(True)
        self.rotation_axis_settings.append(field)
    self.gripper_enabled_setting=Gtk.CheckButton(label='启用夹爪')
    self.gripper_enabled_setting.set_tooltip_text('修改后应用参数；失去追踪或停止不会自动松开夹爪。')
    self.teleop_button_setting=Gtk.ComboBoxText()
    for key,text in (('trigger','食指扳机（按住跟随）'),('b','B（旧 APK 兼容）')):
        self.teleop_button_setting.append(key,text)
    self.teleop_button_setting.set_active_id('trigger')
    self.gripper_button_setting=Gtk.ComboBoxText()
    self.gripper_button_setting.append('b','右手 B（单击切换开合）')
    self.gripper_button_setting.append('a','右手 A（兼容新版输入）')
    self.gripper_button_setting.set_active_id('a')


def build_dashboard(self):
    css=Gtk.CssProvider()
    css.load_from_data(b'''
        window { background: #111827; color: #e5e7eb; }
        label { color: #e5e7eb; } .title { font-size: 22px; font-weight: bold; }
        .hint { color: #9ca3af; } .pending-text { color: #fbbf24; }
        button { padding: 7px 11px; background-image: none; background-color: #263348; border: 1px solid #475569; }
        button:hover { background-color: #334866; } button:disabled { background-color: #1b2535; }
        button:disabled label { color: #7b8799; } frame { border: 1px solid #334155; border-radius: 8px; }
        notebook, notebook > stack, notebook > header, notebook > header tabs tab { background-color: #182337; background-image: none; }
        notebook > header tabs tab:checked { background-color: #334866; }
        menu, menuitem { background-color: #182337; color: #e5e7eb; }
        combobox, combobox cellview { color: #e5e7eb; }
    ''')
    Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(),css,Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    create_controls(self)
    self.workspace_root=vertical()
    self.add(self.workspace_root)
    title=Gtk.Label(label='Quest → xArm7',xalign=0)
    title.get_style_context().add_class('title')
    button_row=pack(vertical(),Gtk.Label(label='遥操按键（修改后应用参数）',xalign=0),self.teleop_button_setting)
    arrange_dashboard(self,self.workspace_root,title,self.receiver_stop,self.diagnostic_zero,self.diagnostic_export,button_row)
    # Home speed keeps its separate path-validation lifecycle.
    fields=list(self.tuning_fields.values())
    fields += [self.gripper_open_setting,self.gripper_close_setting,self.gripper_speed_setting,self.gripper_force_setting]
    for field in fields:
        field.connect('value-changed',self.settings_edited)
    for field in [self.rotation_enabled_setting,self.gripper_enabled_setting,self.input_filter_setting,*self.rotation_axis_settings]:
        field.connect('toggled',self.settings_edited)
    for field in (self.teleop_button_setting,self.gripper_button_setting):
        field.connect('changed',self.settings_edited)

def arrange_dashboard(self, root, title, receiver_stop, zero, export, button_row):
    """Reuse the tested controls/signals; show only daily operations by default."""
    root.set_spacing(8)

    def detach(widget):
        parent = widget.get_parent()
        if parent is not None:
            parent.remove(widget)
        return widget

    def pack(box, widget, expand=False):
        box.pack_start(detach(widget), expand, expand, 0)

    def row(*widgets):
        box = Gtk.Box(spacing=8)
        for widget in widgets:
            pack(box, widget)
        return box

    def label(text):
        return Gtk.Label(label=text, xalign=0)

    def panel(name):
        frame = Gtk.Frame(label=name)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=10)
        frame.add(box)
        return frame, box

    # Create panels with explicit ownership; no dependence on child indices.
    title.set_text("Quest → xArm7")
    hint = label("按住食指扳机跟随 · 松开暂停、调整握姿 · 再按继续（不必重新启用）")
    hint.get_style_context().add_class("hint")
    self.connect_button.set_label("连接手柄")
    self.robot_connect.set_label("连接机械臂")
    self.quest_open.set_label("打开头显输入")
    self.robot_enable.set_label("启用遥操")
    self.robot_release.set_label("禁用遥操")
    self.robot_stop.set_label("停止 [空格]")
    self.robot_stop.get_style_context().add_class("stop-button")
    self.robot_enable.get_style_context().add_class("primary-button")
    toolbar = row(self.connect_button, self.robot_connect, self.quest_open,
                  self.robot_enable, self.robot_release, self.robot_stop)
    state_frame, state_box = panel("连接与遥操")
    pack(state_box, self.robot_status)
    pack(state_box, self.status)

    self.lan_info = label('局域网 HTTPS：点击“打开头显输入”获取地址与配对码' if self.lan_host else '')
    self.lan_info.set_selectable(True)
    self.lan_info.set_line_wrap(True)
    self.lan_info.set_max_width_chars(85)
    self.lan_info.set_no_show_all(not bool(self.lan_host))

    if self.lan_host:
        self.quest_wake.set_sensitive(False)
    self.device_summary = label("头显：等待连接")

    pack(state_box, self.robot_error)
    self.last_stop_note = label('')
    self.last_stop_note.set_line_wrap(True)
    self.last_stop_note.set_max_width_chars(85)

    self.robot_error.get_style_context().add_class("error-text")
    self.robot_error.set_no_show_all(True)
    self.robot_status.get_style_context().add_class("control-state")
    for widget in (self.robot_status, self.status, self.button_status, self.robot_error):
        widget.set_line_wrap(True)
        widget.set_max_width_chars(85)

    mode_frame, mode_box = panel('由操作者选择遥操映射（应用后保持只读）')
    self.mapping_mode_note = label('等待读取后台映射状态')
    self.mapping_mode_note.set_line_wrap(True)
    self.mapping_mode_note.set_max_width_chars(85)
    pack(mode_box, self.mapping_mode_note)
    self.matrix_choice = Gtk.ComboBoxText()
    self.matrix_choice.append('old', '旧方向映射（不用新矩阵）')
    self.matrix_choice.append('new', '新 6×6 矩阵')
    self.matrix_choice.set_active_id('old')
    self.matrix_choice_dirty = False
    self.matrix_choice.connect('changed', self.matrix_choice_changed)
    self.matrix_select_button = Gtk.Button(label='应用我的选择')
    self.matrix_select_button.set_sensitive(False)
    self.matrix_select_button.connect('clicked', self.confirm_matrix_choice)
    pack(mode_box, row(label('我要使用'), self.matrix_choice, self.matrix_select_button))
    self.matrix_choice_note = label('选择框只表示待应用选择；上方显示后台实际使用的映射。')
    self.matrix_choice_note.set_line_wrap(True)
    self.matrix_choice_note.set_max_width_chars(85)
    pack(mode_box, self.matrix_choice_note)
    self.matrix_details = Gtk.Expander(label='查看当前矩阵')
    self.matrix_values = label('等待读取矩阵')
    self.matrix_values.set_selectable(True)
    self.matrix_details.add(self.matrix_values)
    pack(mode_box, self.matrix_details)
    pack(mode_box, self.mapping)

    pose_frame, pose_box = panel("实时位姿")
    self.pose_grid = Gtk.Grid(column_spacing=10, row_spacing=7)
    self.pose_grid.set_tooltip_text("原始手柄与 TCP 使用不同坐标系。欧拉角在组合转动时可能多列同时变化，不能逐列对比；用下方对齐后相对变化检查方向。旋转控制使用连续四元数增量。")
    for col, text in enumerate(("轴", "手柄 m / °", "目标 mm / °", "下发 mm / °", "实测 mm / °")):
        header = label(text)
        header.set_hexpand(col > 0)
        header.get_style_context().add_class("hint")
        self.pose_grid.attach(header, col, 0, 1, 1)
    self.hand_angles = [label("—") for _ in range(3)]
    self.robot_pose_actual = [label("—") for _ in range(6)]
    self.robot_pose_target = [label("—") for _ in range(6)]
    self.robot_pose_command = [label("—") for _ in range(6)]
    for index, axis in enumerate(("X", "Y", "Z", "Rx", "Ry", "Rz")):
        axis_label = label(axis)
        axis_label.set_markup(f'<span foreground="{("#f05052", "#38bf85", "#4294f2")[index % 3]}"><b>{axis}</b></span>')
        self.pose_grid.attach(axis_label, 0, index+1, 1, 1)
        hand_value = self.values[index] if index < 3 else self.hand_angles[index-3]
        for col, widget in enumerate((hand_value, self.robot_pose_target[index], self.robot_pose_command[index], self.robot_pose_actual[index]), 1):
            detach(widget)
            widget.get_style_context().add_class("compact-value")
            widget.set_selectable(True)
            widget.set_hexpand(True)
            self.pose_grid.attach(widget, col, index+1, 1, 1)
    pose_box.pack_start(self.pose_grid, False, False, 0)
    self.aligned_hand_feedback = label("对齐后手柄 ΔXYZ / Δ旋转：—（机器人基座轴，1×）")
    self.aligned_hand_feedback.set_line_wrap(True)
    self.aligned_hand_feedback.set_tooltip_text("相对本次接管/预览起点、未乘倍率的手柄变化。Δ旋转是旋转向量：方向为转轴、长度为转角，不是三个独立欧拉角。1× 时各方向等比例；换采集会话后须先核对方向矩阵。")
    pose_box.pack_start(self.aligned_hand_feedback, False, False, 0)
    self.motion_feedback = label("实速 — · 目标→实测 — · 下发→实测 — · 控制 — Hz")
    self.motion_feedback.set_line_wrap(True)
    self.limit_feedback = label("")
    self.limit_feedback.set_no_show_all(True)
    self.limit_feedback.get_style_context().add_class("hint")
    pose_box.pack_start(self.motion_feedback, False, False, 0)
    pose_box.pack_start(self.limit_feedback, False, False, 0)
    self.ik_feedback_label = label("")
    self.ik_feedback_label.set_line_wrap(True)
    self.ik_feedback_label.set_max_width_chars(85)
    self.ik_feedback_label.set_no_show_all(True)
    self.ik_feedback_label.get_style_context().add_class("hint")
    pose_box.pack_start(self.ik_feedback_label, False, False, 0)

    self.apply_button = Gtk.Button(label="应用并暂停")
    self.apply_button.connect("clicked", self.apply_settings)
    self.responsive_button = Gtk.Button(label="跟手增强")
    self.responsive_button.set_tooltip_text("应用 150 mm/s、60°/s、平移加速度 600、角加速度 240、关节 57.3°/s、平移跟随容差 12 mm，并暂停；保留当前倍率和方向矩阵。")
    self.responsive_button.connect('clicked',self.apply_responsive_settings)
    self.gain_limit_note = label('平移 ≤10× / 旋转 ≤5×')
    self.input_filter_setting = Gtk.CheckButton(label='自适应抑抖')
    self.input_filter_setting.set_tooltip_text('低速增强平滑，运动时自动降低延迟；不改变倍率。修改后点击“应用并暂停”。')
    self.input_filter_feedback = label('输入滤波：等待后台状态')
    self.input_filter_feedback.set_line_wrap(True)
    self.effective_settings_note = label('后台生效参数：等待连接；修改输入框后须点击“应用并暂停”。')
    self.effective_settings_note.set_line_wrap(True)
    self.effective_settings_note.set_max_width_chars(85)
    self.tuning_fields = dict(speed_mm_s=self.speed_setting, translation_scale=self.scale_setting,
        joint_speed_rad_s=self.joint_speed_setting, acceleration_mm_s2=self.acceleration_setting,
        tracking_lead_mm=self.tracking_lead_setting, rotation_scale=self.rotation_scale_setting,
        rotation_speed_deg_s=self.rotation_speed_setting,
        rotation_acceleration_deg_s2=self.rotation_acceleration_setting)
    self.gripper_open_button.set_label("打开夹爪")
    self.gripper_close_button.set_label("闭合夹爪")
    self.gripper_enabled_setting.set_label("启用夹爪")
    self.gripper_summary = label("夹爪 — mm · B 单击切换开合 · 松开保持")

    self.home_save_button = Gtk.Button(label="记录初始 Pose")
    self.home_check_button = Gtk.Button(label="检查 Pose 复位")
    self.home_check_button.set_tooltip_text("只读：以保存的 XYZ/RxRyRz 为目标求解当前分支，并检查运动路径，不移动。")
    self.home_start_button = Gtk.Button(label="恢复初始 Pose")
    for button, command in ((self.home_save_button, "home_save"), (self.home_check_button, "home_check"),
                            (self.home_start_button, "home_start")):
        button.connect("clicked", lambda _button, cmd=command: self.recovery_action(cmd))
        button.set_sensitive(False)
    self.home_speed_setting = Gtk.SpinButton.new_with_range(*HOME_SPEED_LIMITS, 1)
    self.home_speed_setting.set_value(HOME_SPEED_DEFAULT)
    self.home_speed_setting.set_tooltip_text('初始位姿复位的关节速度，5–30°/s；加速度自动为速度的 2 倍。修改后重新检查路径。重开界面默认 5°/s。')
    self.home_speed_setting.connect('value-changed', self.home_speed_changed)
    self.home_feedback = label("未设置初始位姿；异常姿态可展开“关节恢复”，先低速调整")
    self.home_feedback.set_line_wrap(True)
    self.home_feedback.set_max_width_chars(85)
    self.recovery_panel = Gtk.Expander(label="关节恢复 / 初始位姿详情")
    recovery_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=8)
    self.recovery_panel.add(recovery_box)
    self.recovery_joint = Gtk.ComboBoxText()
    for axis in range(7):
        self.recovery_joint.append(str(axis), f"J{axis+1}")
    self.recovery_joint.set_active_id("0")
    self.recovery_step = Gtk.SpinButton.new_with_range(.5, 5, .5)
    self.recovery_step.set_value(2)
    self.joint_minus = Gtk.Button(label="负方向小步恢复")
    self.joint_plus = Gtk.Button(label="正方向小步恢复")
    self.joint_minus.connect("clicked", lambda *_: self.recovery_action("joint_recover", -1))
    self.joint_plus.connect("clicked", lambda *_: self.recovery_action("joint_recover", 1))
    recovery_box.pack_start(row(label("关节"), self.recovery_joint, label("步长 °"), self.recovery_step,
                               self.joint_minus, self.joint_plus), False, False, 0)
    self.restore_gripper = Gtk.CheckButton(label="返回初始位姿后恢复记录的夹爪开度（会开/合，默认关闭）")
    recovery_box.pack_start(self.restore_gripper, False, False, 0)
    self.home_details = label("初始点未记录")
    self.home_details.set_line_wrap(True)
    self.home_details.set_selectable(True)
    recovery_box.pack_start(self.home_details, False, False, 0)
    safety_note = label("先禁用遥操。关节恢复不依赖 IK，但只检查关节限位/正解，不判断环境碰撞。\n"
                        "确认整条路径净空、急停可触达后才能执行；错误/急停不自动清除。\n"
                        "当前关节已越出硬件限位时禁止自动恢复，请用厂家示教/故障处理流程。")
    safety_note.set_line_wrap(True)
    recovery_box.pack_start(safety_note, False, False, 0)

    self.advanced_panel = Gtk.Expander(label="更多参数 / 按键设置")
    advanced_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=8)
    self.advanced_panel.add(advanced_box)
    pack(advanced_box, button_row)
    pack(advanced_box,field_grid([
        ('关节速度 °/s',self.joint_speed_setting),('平移加速度 mm/s²',self.acceleration_setting),
        ('跟随容差上限 mm',self.tracking_lead_setting),('角加速度 °/s²',self.rotation_acceleration_setting),
        ('夹爪速度 mm/s',self.gripper_speed_setting),
        ('张开 mm',self.gripper_open_setting),('闭合 mm',self.gripper_close_setting),
        ('夹爪手柄键',self.gripper_button_setting)]))
    tuning_help = label('平移优先低延迟：正常速度内按倍率跟随，超出机械臂能力时压缩多余位移，避免长时间追赶。\n倍率越大，需要的速度越高。松开扳机取消跟随；掉线、关节、IK 连续性及硬件故障保护保留。')
    tuning_help.set_line_wrap(True)
    pack(advanced_box, tuning_help)
    pack(advanced_box,row(label('基座旋转轴'),*self.rotation_axis_settings))
    pack(advanced_box, self.range_status)
    pack(advanced_box, self.effective_settings_note)
    pack(advanced_box, self.gain_limit_note)
    pack(advanced_box, self.robot_takeover)

    self.diagnostics_panel = Gtk.Expander(label="曲线 / 诊断 / 导出")
    diagnostics_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=8)
    self.diagnostics_panel.add(diagnostics_box)
    pack(diagnostics_box, row(zero, export))
    for widget in (*self.deltas, self.orientation, self.counters, self.movement,
                   self.robot_tcp, self.robot_rotation, self.robot_stats, self.speed_feedback,
                   self.gripper_status, self.instructions):
        widget.set_line_wrap(True)
        widget.set_max_width_chars(85)
        pack(diagnostics_box, widget)
    pack(diagnostics_box, self.plot)
    # The shell is explicit: header never scrolls, pages own their own content.
    hint.set_text('扳机按住跟随 · 松开暂停 · B 单击切换夹爪')
    title_row=row(title)
    pack(title_row,hint,True)
    for widget in (title_row,toolbar,state_frame):
        root.pack_start(widget,False,False,0)
    self.header_status_box=state_box
    self.workspace_notebook=Gtk.Notebook()
    self.workspace_notebook.set_scrollable(True)
    root.pack_start(self.workspace_notebook,True,True,0)

    # Teleoperation: observation and recovery on the left; parameters on the right.
    left=vertical(margin=8)
    right=vertical(margin=8)
    pack(pose_box,self.button_status)
    pack(left,pose_frame)
    home_frame,home_box=panel('初始位姿 / 复位')
    pack(home_box,row(self.home_check_button,self.home_start_button))
    pack(home_box,row(label('复位 °/s'),self.home_speed_setting,self.home_save_button))
    pack(home_box,self.home_feedback)
    pack(home_box,self.recovery_panel)
    pack(left,home_frame)
    history=Gtk.Expander(label='上次停止原因')
    history.add(detach(self.last_stop_note))
    pack(left,history)

    settings_frame,settings_box=panel('运动参数')
    settings_box.pack_start(field_grid([
        ('平移速度 mm/s',self.speed_setting),('平移倍率 ×',self.scale_setting),
        ('旋转速度 °/s',self.rotation_speed_setting),('旋转倍率 ×',self.rotation_scale_setting)]),False,False,0)
    pack(settings_box,self.rotation_enabled_setting)
    pack(settings_box,self.input_filter_setting)
    pack(settings_box,self.input_filter_feedback)
    pack(settings_box,row(self.apply_button,self.responsive_button))
    self.settings_edit_note=label('修改参数后点击“应用并暂停”；生效值见更多参数。')
    self.settings_edit_note.set_line_wrap(True)
    self.settings_edit_note.set_max_width_chars(36)
    pack(settings_box,self.settings_edit_note)
    pack(settings_box,self.advanced_panel)
    for widget in (self.input_filter_feedback,self.effective_settings_note,self.range_status,tuning_help):
        widget.set_max_width_chars(30)
        widget.set_line_wrap(True)
    pack(right,settings_frame)
    gripper_frame,gripper_box=panel('夹爪')
    pack(gripper_box,self.gripper_enabled_setting)
    self.gripper_force_setting.set_tooltip_text('G2 力度设定：1–100%，每步 1%。修改后点击上方“应用并暂停”；下次开合使用新力度。')
    pack(gripper_box,field_grid([('抓取力度 %',self.gripper_force_setting)]))
    force_note=label('1–100% 力度设定，非实测力。修改后点击上方“应用并暂停”，下次开合生效。')
    force_note.set_line_wrap(True)
    force_note.set_max_width_chars(30)
    pack(gripper_box,force_note)
    pack(gripper_box,row(self.gripper_open_button,self.gripper_close_button))
    self.gripper_summary.set_line_wrap(True)
    self.gripper_summary.set_max_width_chars(36)
    pack(gripper_box,self.gripper_summary)
    pack(right,gripper_frame)
    teleop=Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
    teleop.pack1(scroll_page(left),True,False)
    teleop.pack2(scroll_page(right),True,False)
    teleop.set_position(670)
    self.teleop_page=teleop
    self.workspace_notebook.append_page(teleop,Gtk.Label(label='遥操控制'))

    # Device actions and full diagnostics remain available, without crowding daily work.
    device=vertical(margin=12)
    device_frame,device_box=panel('头显连接 / 设备维护')
    pack(device_box,self.device_summary)
    pack(device_box,self.lan_info)
    pack(device_box,row(self.quest_restart,self.quest_wake,self.quest_inspect,receiver_stop))
    for widget in (self.quest_app_status,self.quest_app_note,self.detail):
        pack(device_box,widget)
    pack(device,device_frame)
    pack(device,mode_frame)
    pack(device,self.diagnostics_panel)
    self.device_page=scroll_page(device)
    # RecordingPanel adds collection/library pages before this fourth page.

    for expander in (self.recovery_panel, self.advanced_panel, self.diagnostics_panel):
        expander.set_expanded(False)
    compact_css = Gtk.CssProvider()
    compact_css.load_from_data(b".compact-value { font-family: monospace; font-size: 17px; } .control-state { font-size: 16px; font-weight: bold; } .stop-button { background-color: #8b2434; border-color: #dc4b5b; } .primary-button { background-color: #1d4d70; } .error-text { color: #ff8c8c; } button { padding: 7px 11px; } frame { border-radius: 8px; }")
    Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), compact_css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION+1)

