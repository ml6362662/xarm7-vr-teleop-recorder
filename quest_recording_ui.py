"""GTK collection page; starts the passive recorder, never a second robot driver."""
import json
from pathlib import Path
import subprocess
import threading
import time

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, GLib, GdkPixbuf, Gdk
from quest_ui_state import recording_actions, capture_summary, process_running, fresh_timestamp

ROOT = Path(__file__).resolve().parent


class RecordingPanel:
    def __init__(self, owner, root):
        self.owner = owner
        self.process = None
        self.export_process = None
        self.export_result = None
        self.quality_report = None
        self.export_scan_only = False
        self.report_dialog = None
        self.snapshot = {}
        self.pending = False
        self.closed = False
        self.review_requested = False
        self.review_dialog_open = False
        self.review_dialog = None
        self.review_keep = True
        config = json.loads((ROOT/"config.json").read_text())
        self.config = config
        self.preview_cache = {}
        from quest_recording_layout import build_recording_page
        build_recording_page(self,owner)

    def button(self,text,callback):
        widget=Gtk.Button(label=text)
        widget.connect("clicked",callback)
        self.controls.pack_start(widget,False,False,0)
        return widget

    def dataset_root(self):
        name=self.dataset.get_text().strip()
        if not name or not name.replace("_","").replace("-","").isalnum():
            raise ValueError("数据集名称只能包含字母、数字、下划线和连字符")
        return Path(self.folder.get_text()).expanduser().resolve()/name

    def action_state(self):
        return recording_actions(self.snapshot, running=process_running(self.process),
            exporting=process_running(self.export_process), pending=self.pending,
            demo=self.owner.demo, closed=self.closed, now=time.monotonic())

    def update_buttons(self):
        state=self.action_state()
        for widget,enabled in ((self.prepare,state.prepare),(self.start,state.start),
                (self.save,state.review),(self.discard,state.review),(self.stop,state.stop)):
            widget.set_sensitive(enabled)
        for widget in (self.format,self.export_fps,self.clean_export,self.include_failures):
            widget.set_sensitive(not self.closed and not process_running(self.export_process))
        for widget in (self.dataset,self.folder):
            widget.set_sensitive(state.configure)
        self.task.set_sensitive(state.edit_task)
        self.start.set_tooltip_text(state.reason or '开启本条后，只记录有效遥操')
        self.update_home_buttons()
        if hasattr(self, 'library'):
            self.library.update_action()
        else:
            self.export.set_sensitive(False)
            self.scan.set_sensitive(False)

    def update_home_buttons(self):
        self.home_speed.set_sensitive(self.owner.home_speed_setting.get_sensitive())
        for command,button in self.home_buttons.items():
            source={'release':self.owner.robot_release,'home_save':self.owner.home_save_button,
                    'home_check':self.owner.home_check_button,'home_start':self.owner.home_start_button}[command]
            button.set_sensitive(source.get_sensitive())
        self.home_status.set_text(self.owner.home_feedback.get_text()+'\n'+self.owner.home_details.get_text())

    def home_action(self,command):
        if command=='release':
            self.owner.robot_command('release')
        else:
            self.home_speed.update()
            self.owner.recovery_action(command)

    def start_process(self,*_):
        if not self.action_state().prepare:
            return
        try:
            root=self.dataset_root()
            python=Path(self.config["lerobot_root"])/".venv/bin/python"
            self.process=subprocess.Popen([str(python),"-u",str(ROOT/"quest_recorder.py"),"--root",str(root)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            self.snapshot={}
            self.clear_previews()
            self.banner.set_text("数据采集：正在准备；不会使能或移动机械臂")
            threading.Thread(target=self.read_output,args=(self.process,False),daemon=True).start()
        except (ValueError,OSError) as error:
            self.detail.set_text(str(error))
        self.update_buttons()

    def read_output(self,process,export):
        tail=[]
        for line in process.stdout:
            tail.append(line.strip())
            tail=tail[-12:]
            try:
                item=json.loads(line)
            except ValueError:
                continue
            if not isinstance(item,dict):
                continue
            if export and item.get('type')=='export_result':
                GLib.idle_add(self.received_export,process,item)
            if item.get("type")=="recorder_status":
                GLib.idle_add(self.received,process,item)
        code=process.wait()
        GLib.idle_add(self.finished,process,export,code,"\n".join(tail))

    def received(self,process,item):
        if self.closed or process is not self.process:
            return False
        self.snapshot=item
        self.pending=False
        state=item.get('capture_state','idle')
        if state!='review' and self.review_dialog is not None:
            self.review_dialog.response(Gtk.ResponseType.CANCEL)
        self.banner.set_text(capture_summary(item))
        details=[item.get("ready_error","") or "设备就绪",item.get('message',''),
                 f"机器人样本 {item.get('samples',0)} · 丢包 {item.get('telemetry_gaps',0)}"]
        interruption = item.get('last_input_pause')
        if interruption:
            details.append('最近输入中断：' + interruption['reason'] +
                           ('（已重新接管）' if interruption.get('resumed_s') is not None else '（尚未重新接管）'))
        for name,camera_state in item.get("cameras",{}).items():
            details.append(f"{name}: {camera_state.get('measured_fps',0):.1f} FPS · 丢帧 {camera_state.get('dropped',0)} · {camera_state.get('error','')}")
        if item.get("last_episode"):
            details.append(item["last_episode"])
        self.detail.set_text("\n".join(details))
        self.update_buttons()
        if state=='review' and self.review_requested and not self.review_dialog_open:
            self.review_requested=False
            GLib.idle_add(self.review_episode)
        return False

    def finished(self,process,export,code,tail):
        if self.closed:
            return False
        if export and process is self.export_process:
            if self.export_result:
                counts=self.export_result['summary']
                prefix=('检查完成' if self.export_scan_only else '导出完成') if code==0 else '导出失败'
                self.export_status.set_text(f"{prefix} · 合格 {counts['accepted']} · 不合格 {counts['rejected']} · 未纳入 {counts['excluded']} · 有警告 {counts.get('warnings',0)}\n"+
                    str(self.export_result.get('output') or self.export_result.get('report')))
            else:
                self.export_status.set_text(('检查/导出完成：' if code==0 else '检查/导出失败：')+tail)
        elif process is self.process:
            if self.review_dialog is not None:
                self.review_dialog.response(Gtk.ResponseType.CANCEL)
            self.snapshot={}
            self.clear_previews()
            self.pending=False
            self.banner.set_text("数据采集：已关闭" if code==0 else "数据采集：进程异常退出；检查未完成记录")
            if code:
                self.detail.set_text(tail)
        self.update_buttons()
        return False

    def send(self,command,**values):
        if not self.process or self.process.poll() is not None or self.pending:
            return
        actions=self.action_state()
        if (command=='start' and not actions.start) or (command=='review' and not actions.review):
            return
        try:
            self.process.stdin.write(json.dumps(dict(command=command,**values),ensure_ascii=False)+"\n")
            self.process.stdin.flush()
            self.pending=True
        except (OSError,ValueError) as error:
            self.detail.set_text(str(error))
        self.update_buttons()

    def save_episode(self,*_,keep=True):
        self.review_keep=keep
        if self.snapshot.get('capture_state')=='review':
            self.review_episode()
        else:
            self.review_requested=True
            self.send('review')

    def review_episode(self):
        if self.closed or self.review_dialog_open or self.snapshot.get('capture_state')!='review':
            return False
        self.review_dialog_open=True
        review_process, review_episode_id = self.process, self.snapshot.get('episode')
        dialog=Gtk.Dialog(title='本条轨迹：人工判定与保留',transient_for=self.owner,modal=False)
        self.review_dialog=dialog
        dialog.connect('key-press-event',self.review_key_pressed)
        dialog.add_button('稍后判定（保持冻结）',Gtk.ResponseType.CANCEL)
        dialog.add_button('确认结果',Gtk.ResponseType.OK)
        box=dialog.get_content_area()
        box.set_spacing(10)
        box.set_border_width(16)
        note=Gtk.Label(label='请根据刚才的完整操作判定结果。\n成功标准：已完成本条任务描述中的目标操作。\n选择失败也可以保留用于分析或后续训练。',xalign=0)
        box.add(note)
        outcome=Gtk.ComboBoxText()
        outcome.append('success','成功')
        outcome.append('failure','失败')
        outcome.set_active(-1)
        box.add(outcome)
        keep=Gtk.CheckButton(label='保留本条（取消勾选则标记丢弃，不进入导出）')
        keep.set_active(self.review_keep)
        box.add(keep)
        dialog.set_response_sensitive(Gtk.ResponseType.OK,False)
        outcome.connect('changed',lambda *_:dialog.set_response_sensitive(Gtk.ResponseType.OK,outcome.get_active_id() is not None))
        def responded(_dialog,response):
            try:
                if (not self.closed and self.process is review_process and
                        self.snapshot.get('episode')==review_episode_id and self.snapshot.get('capture_state')=='review'
                        and response==Gtk.ResponseType.OK and outcome.get_active_id() is not None):
                    self.send('save' if keep.get_active() else 'discard',success=outcome.get_active_id()=='success')
            finally:
                self.review_dialog=None
                self.review_dialog_open=False
                dialog.destroy()
        dialog.connect('response',responded)
        dialog.show_all()
        return False

    def review_key_pressed(self,dialog,event):
        if event.keyval==Gdk.KEY_Escape:
            dialog.response(Gtk.ResponseType.CANCEL)
            return True
        return False

    def shutdown(self,*_):
        # SIGTERM requests flush + incomplete marker, not deletion or robot motion.
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def close(self):
        if self.report_dialog is not None:
            self.report_dialog.response(Gtk.ResponseType.CLOSE)
        self.closed=True
        if self.review_dialog is not None:
            self.review_dialog.response(Gtk.ResponseType.CANCEL)
        self.library.close()
        GLib.source_remove(self.timer)
        self.shutdown()

    def clear_previews(self):
        self.preview_cache.clear()
        for image in self.images.values():
            image.clear()

    def refresh(self):
        if self.closed or self.owner.closing:
            return False
        self.update_buttons()
        fresh=fresh_timestamp(self.snapshot.get("monotonic_s"),time.monotonic(),2)
        if self.snapshot and not fresh:
            self.banner.set_text("数据采集：状态超时，请检查采集进程；遥操控制独立运行")
            self.clear_previews()
        if process_running(self.process) and fresh:
            directory=self.snapshot.get("preview_dir")
            if directory:
                for name,image in self.images.items():
                    path=Path(directory)/(name+".jpg")
                    try:
                        stat=path.stat()
                        signature=(str(path),stat.st_mtime_ns,stat.st_size)
                        if self.preview_cache.get(name)==signature:
                            continue
                        pix=GdkPixbuf.Pixbuf.new_from_file_at_scale(str(path),400,300,True)
                        image.set_from_pixbuf(pix)
                        self.preview_cache[name]=signature
                    except (OSError,GLib.Error):
                        image.clear()
                        self.preview_cache.pop(name,None)
        return True

    def received_export(self,process,item):
        if not self.closed and process is self.export_process:
            self.export_result=item
            self.report_button.set_sensitive(True)
            if hasattr(self.library,'set_quality_report'):
                def apply(report,error):
                    if self.closed or process is not self.export_process or self.export_result is not item:
                        return
                    if error:
                        self.export_status.set_text('质量报告读取失败：'+error)
                        return
                    self.quality_report=report
                    self.library.set_quality_report(report)
                from quest_quality_model import read_report
                self.library.submit(lambda:read_report(item['report']),apply)
        return False

    def show_quality_report(self,*_):
        if not self.export_result:
            return
        from quest_quality_model import read_report
        from quest_quality_ui import show_report
        if self.report_dialog is not None:
            self.report_dialog.present()
            return
        result=self.export_result
        def apply(report,error):
            if self.closed or self.export_result is not result:
                return
            if error:
                self.export_status.set_text('报告读取失败：'+error)
                return
            self.quality_report=report
            self.library.set_quality_report(report)
            show_report(self,report)
        self.library.submit(lambda:read_report(result['report']),apply)

    def start_export(self,*_,scan_only=False):
        if not self.action_state().export or self.library.busy:
            return
        try:
            root=getattr(self.library,'root',None)
            if root is None:
                raise ValueError('请先在数据管理页选择原始数据集')
            kind=self.format.get_active_id()
            from uuid import uuid4
            output=root/('quality_reports' if scan_only else 'exports')/(
                ('check' if scan_only else kind)+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
            python=(str(Path(self.config['lerobot_root'])/'.venv/bin/python') if scan_only or kind=='lerobot21' else self.config['openteach_python'])
            command=[python,'-u',str(ROOT/'export_recording.py'),str(root),str(output),'--format',kind,
                     '--fps',str(self.export_fps.get_value_as_int())]
            if self.include_failures.get_active():
                command.append('--include-failures')
            if scan_only:
                command.append('--scan-only')
            elif self.clean_export.get_active():
                command.append('--clean')
            if kind=='smooth269':
                command=[str(Path(self.config['lerobot_root'])/'.venv/bin/python'),'-B','-u','/home/a/xarm_export269/export269.py',str(root),str(output)]
                if scan_only:command.append('--scan-only')
            self.export_result=None
            self.quality_report=None
            if hasattr(self.library,'set_quality_report'):
                self.library.set_quality_report(None)
            self.export_scan_only=scan_only
            self.report_button.set_sensitive(False)
            if self.report_dialog is not None:
                self.report_dialog.response(Gtk.ResponseType.CLOSE)
            self.export_process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.export_status.set_text(('正在后台检查：' if scan_only else '正在后台导出：')+str(root))
            threading.Thread(target=self.read_output,args=(self.export_process,True),daemon=True).start()
            self.update_buttons()
        except (ValueError,OSError) as error:
            self.export_status.set_text(str(error))
