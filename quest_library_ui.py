"""Compact, lazy-loaded raw data library. Device access is never performed here."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk, GLib, GdkPixbuf, Pango

import recording_library as library
from quest_ui_state import process_running
from quest_quality_model import STATES, severity, reasons, matches, counts


LABELS = dict(success='成功', failure='失败', unlabeled='未标注', incomplete='未完成',
              discarded='已丢弃', recording='录制中', invalid='需检查')


class LibraryPanel:
    def __init__(self, recording):
        self.recording = recording
        self.closed = False
        self.updating = False
        self.busy = False
        self.generation = 0
        self.rows = []
        self.selected = None
        self.root = None
        self.preview_running = False
        self.desired_preview = None
        self.play_timer = None
        self.quality_report = None
        self.reveal_target = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='data-library')
        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
        bar = Gtk.Box(spacing=8)
        self.datasets = Gtk.ComboBoxText()
        self.datasets.connect('changed', self.dataset_changed)
        self.filter = Gtk.ComboBoxText()
        for key, label in [('kept','已保留'),('review','待整理'),('all','全部'),('trash','回收站')]:
            self.filter.append(key, label)
        self.filter.set_active_id('kept')
        self.filter.connect('changed', self.dataset_changed)
        self.refresh_button = Gtk.Button(label='刷新')
        self.refresh_button.connect('clicked', self.refresh)
        bar.pack_start(Gtk.Label(label='数据集'), False, False, 0)
        bar.pack_start(self.datasets, True, True, 0)
        bar.pack_start(self.filter, False, False, 0)
        bar.pack_start(self.refresh_button, False, False, 0)
        self.widget.pack_start(bar, False, False, 0)
        self.summary = Gtk.Label(label='选择数据集后查看轨迹；回收站可恢复，不会永久删除。', xalign=0)
        self.summary.set_line_wrap(True)
        self.widget.pack_start(self.summary, False, False, 0)
        self.quality_summary=Gtk.Label(label='质量：未检查 · 点击“检查数据”查看每条轨迹的问题。',xalign=0)
        self.quality_summary.set_line_wrap(True)
        self.widget.pack_start(self.quality_summary,False,False,0)
        self.widget.pack_start(recording.export_box, False, False, 0)
        columns=Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.widget.pack_start(columns,True,True,0)
        listing=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=8)
        viewing=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=8)
        columns.pack1(listing,True,False)
        preview_scroll=Gtk.ScrolledWindow()
        preview_scroll.set_policy(Gtk.PolicyType.AUTOMATIC,Gtk.PolicyType.AUTOMATIC)
        preview_scroll.add(viewing)
        columns.pack2(preview_scroll,True,False)
        columns.set_position(440)
        self.store = Gtk.ListStore(str, str, str, str, str, str, str, str)
        self.tree = Gtk.TreeView(model=self.store)
        for i, title in enumerate(['采集时间','结果','有效时长','占用','任务']):
            cell = Gtk.CellRendererText()
            if i == 4:
                cell.set_property('ellipsize', Pango.EllipsizeMode.END)
            column = Gtk.TreeViewColumn(title, cell, text=i)
            if i == 4:
                column.set_expand(True)
                column.set_max_width(160)
            self.tree.append_column(column)
        quality_cell=Gtk.CellRendererText()
        self.tree.insert_column(Gtk.TreeViewColumn('质量',quality_cell,text=6,foreground=7),2)
        self.tree.get_selection().connect('changed', self.selection_changed)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_min_content_height(190)
        scroll.add(self.tree)
        listing.pack_start(scroll, True, True, 0)
        self.details = Gtk.Label(label='选中一条轨迹，拖动时间轴查看双相机画面。', xalign=0)
        self.details.set_line_wrap(True)
        self.details.set_max_width_chars(90)
        self.details.set_selectable(True)
        viewing.pack_start(self.details, False, False, 0)
        self.quality_detail=Gtk.Label(xalign=0)
        self.quality_detail.set_line_wrap(True);self.quality_detail.set_max_width_chars(60)
        self.quality_detail.set_selectable(True)
        viewing.pack_start(self.quality_detail,False,False,0)
        images = Gtk.Box(spacing=8)
        self.images = []
        for label in ('腕部', '全局'):
            frame = Gtk.Frame(label=label)
            image = Gtk.Image()
            image.set_size_request(240, 180)
            frame.add(image)
            images.pack_start(frame, True, True, 0)
            self.images.append((frame, image))
        viewing.pack_start(images, False, False, 0)
        timeline = Gtk.Box(spacing=8)
        self.play = Gtk.Button(label='播放')
        self.play.connect('clicked', self.toggle_play)
        self.slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, .1)
        self.slider.set_draw_value(False)
        self.slider.connect('value-changed', self.seek)
        self.time_label = Gtk.Label(label='0.0 / 0.0 s')
        timeline.pack_start(self.play, False, False, 0)
        timeline.pack_start(self.slider, True, True, 0)
        timeline.pack_start(self.time_label, False, False, 0)
        viewing.pack_start(timeline, False, False, 0)
        self.frame_info = Gtk.Label(xalign=0)
        self.frame_info.set_line_wrap(True)
        viewing.pack_start(self.frame_info, False, False, 0)
        actions = Gtk.Box(spacing=8)
        self.move = Gtk.Button(label='移入回收站')
        self.move.connect('clicked', self.move_selected)
        actions.pack_start(self.move, False, False, 0)
        self.notice = Gtk.Label(label='关闭采集器后可整理；回收站不释放磁盘空间。', xalign=0)
        self.notice.set_line_wrap(True)
        actions.pack_start(self.notice, True, True, 0)
        listing.pack_start(actions, False, False, 0)
        self.page = Gtk.ScrolledWindow()
        self.page.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.page.add(self.widget)
        recording.notebook.append_page(self.page, Gtk.Label(label='数据管理'))
        recording.notebook.connect('switch-page', self.switched)
        self.update_action()

    def submit(self, work, apply):
        future = self.pool.submit(work)
        def finished(future):
            try:
                value, error = future.result(), None
            except Exception as exc:
                value, error = None, str(exc)
            def deliver():
                if not self.closed:
                    apply(value, error)
                return False
            GLib.idle_add(deliver)
        future.add_done_callback(finished)

    def switched(self, notebook, page, index):
        if page is self.page:
            if self.reveal_target is None:
                self.refresh()
        else:
            self.stop_play()

    def set_busy(self, value):
        self.busy = value
        for widget in (self.datasets, self.filter, self.refresh_button):
            widget.set_sensitive(not value)
        self.update_action()

    def refresh(self, *_):
        if self.closed or self.busy:
            return
        self.stop_play()
        self.set_busy(True)
        self.generation += 1
        self.desired_preview = None
        self.root = None
        self.update_quality()
        parent = Path(self.recording.folder.get_text()).expanduser()
        selected = self.datasets.get_active_id() or self.recording.dataset.get_text()
        def apply(names, error):
            self.set_busy(False)
            if error:
                self.notice.set_text(error)
                return
            self.parent = parent
            self.updating = True
            self.datasets.remove_all()
            for name in names:
                self.datasets.append(name, name)
            if names:
                self.datasets.set_active_id(selected if selected in names else names[0])
            self.updating = False
            self.dataset_changed()
        self.submit(lambda: library.list_datasets(parent), apply)

    def dataset_changed(self, *_):
        if self.closed or self.updating or self.busy:
            return
        self.stop_play()
        self.generation += 1
        self.desired_preview = None
        self.selected = None
        self.rows = []
        self.store.clear()
        self.clear_preview()
        name = self.datasets.get_active_id()
        if not name:
            self.root = None
            self.update_quality()
            self.summary.set_text('此目录暂无原始数据集；先在采集页选择父目录。')
            self.update_action()
            return
        self.root = None
        root = self.parent/name
        self.update_quality()
        trash = self.filter.get_active_id() == 'trash'
        self.set_busy(True)
        def apply(rows, error):
            self.set_busy(False)
            if error:
                self.notice.set_text(error)
                self.reveal_target=None
                return
            self.root = root
            self.rows = rows
            mode = self.filter.get_active_id()
            kept = {'success', 'failure', 'unlabeled'}
            visible = [r for r in rows if mode in ('all','trash') or
                       (r['category'] in kept if mode == 'kept' else r['category'] in ('incomplete','discarded','invalid'))]
            for row in visible:
                meta = row['metadata']
                try:
                    when = datetime.strptime(row['id'][:15], '%Y%m%d_%H%M%S').strftime('%m-%d %H:%M:%S')
                except ValueError:
                    when = row['id']
                self.store.append([when, LABELS[row['category']], f"{row['duration']:.1f} s",
                    f"{row['bytes']/1024**2:.1f} MB", str(meta.get('task','')), row['id'],'未检查','#aab4c4'])
            self.summary.set_text((f'回收站 {len(rows)} 条' if trash else
                f"已保留 {sum(r['category'] in kept for r in rows)} 条 · 待整理 {sum(r['category'] in ('incomplete','discarded','invalid') for r in rows)} 条") + f' · 当前显示 {len(visible)} 条')
            self.update_quality()
            target=self.reveal_target
            self.reveal_target=None
            index=next((i for i,row in enumerate(visible) if target==(root,row['id'])),None)
            if target and index is None:
                self.notice.set_text('报告中的轨迹已不在当前原始列表，可能已移入回收站或发生变更。')
            if visible and (target is None or index is not None):
                self.tree.get_selection().select_path(Gtk.TreePath.new_from_indices([index or 0]))
                if index is not None:self.tree.scroll_to_cell(Gtk.TreePath.new_from_indices([index]),None,True,.5,0)
            self.update_action()
        self.submit(lambda: library.list_episodes(root, trash), apply)

    def clear_preview(self):
        for _, image in self.images:
            image.clear()
        self.details.set_text('选中一条轨迹，拖动时间轴查看双相机画面。')
        self.frame_info.set_text('')
        self.time_label.set_text('0.0 / 0.0 s')
        self.quality_detail.set_text('')

    def selection_changed(self, selection):
        self.stop_play()
        self.generation += 1
        self.desired_preview = None
        model, iterator = selection.get_selected()
        self.selected = next((r for r in self.rows if iterator is not None and r['id']==model[iterator][5]), None)
        self.clear_preview()
        if self.selected:
            row = self.selected
            meta = row['metadata']
            reason = meta.get('reason') or row.get('error')
            reason = {'operator_save':'人工判定后保存', 'operator_discard':'人工选择丢弃'}.get(reason, reason)
            self.details.set_text(f"{row['id']} · {LABELS[row['category']]} · {row['samples']} 个样本\n" +
                str(meta.get('task','')) + ('\n原因：'+str(reason) if reason else ''))
            self.updating = True
            self.slider.set_range(0, max(.1, row['duration']))
            self.slider.set_value(0)
            self.updating = False
            self.seek()
        self.update_quality_detail()
        self.update_action()

    def set_quality_report(self,report):
        self.quality_report=report
        self.update_quality()

    def current_quality(self):
        if self.quality_report and matches(self.quality_report,self.root) and self.filter.get_active_id()!='trash':
            return {item['episode_id']:item for item in self.quality_report['episodes']}
        return {}

    def update_quality(self):
        indexed=self.current_quality()
        for row in self.store:
            row[6],row[7]=STATES[severity(indexed[row[5]])] if row[5] in indexed else ('未检查','#aab4c4')
        if self.quality_report and matches(self.quality_report,self.root):
            c=counts(self.quality_report)
            self.quality_summary.set_text(f"本次检查 · 不合格 {c['rejected']} · 需复核 {c['warning']} · 通过 {c['accepted']} · 未纳入 {c['excluded']} · {self.quality_report.get('fps','—')} FPS（报告快照，数据变更后请重查）")
        else:
            self.quality_summary.set_text('质量：未检查 · 点击“检查数据”查看每条轨迹的问题。')
        self.update_quality_detail()

    def update_quality_detail(self):
        item=self.current_quality().get(self.selected['id']) if self.selected else None
        self.quality_detail.set_text(('本次检查：'+STATES[severity(item)][0]+'\n'+reasons(item)) if item else '')

    def reveal_episode(self,source,episode_id):
        if self.closed or self.busy:
            return False,'数据列表正在读取，请稍后重试。'
        if not matches({'source':source},self.root):
            return False,'请先在数据管理页选择报告对应的数据集，再点击回看。'
        self.reveal_target=(self.root,episode_id)
        self.recording.notebook.set_current_page(self.recording.notebook.page_num(self.page))
        self.updating=True
        self.filter.set_active_id('all')
        self.updating=False
        self.dataset_changed()
        return True,''

    def seek(self, *_):
        if self.updating or not self.selected or self.closed:
            return
        elapsed = self.slider.get_value()
        self.time_label.set_text(f"{elapsed:.1f} / {self.selected['duration']:.1f} s")
        self.desired_preview = (self.generation, self.root, self.selected['id'], elapsed, self.selected['trash'])
        self.request_preview()

    def request_preview(self):
        if self.preview_running or self.desired_preview is None or self.closed:
            return
        request = self.desired_preview
        self.preview_running = True
        def apply(value, error):
            self.preview_running = False
            if request != self.desired_preview:
                self.request_preview()
                return
            if error:
                self.stop_play()
                for _, image in self.images:
                    image.clear()
                self.frame_info.set_text(error)
                return
            for frame, image in self.images:
                image.clear()
            for (frame, image), data in zip(self.images, value['frames']):
                frame.set_label({'wrist':'腕部','global':'全局'}.get(data['name'], data['name']))
                if data['png']:
                    try:
                        loader = GdkPixbuf.PixbufLoader.new_with_type('png')
                        loader.write(data['png'])
                        loader.close()
                        pix = loader.get_pixbuf()
                        angle = data['rotation']
                        if angle:
                            pix = pix.rotate_simple({90:GdkPixbuf.PixbufRotation.CLOCKWISE,180:GdkPixbuf.PixbufRotation.UPSIDEDOWN,270:GdkPixbuf.PixbufRotation.COUNTERCLOCKWISE}[angle])
                        ratio = min(280/pix.get_width(), 210/pix.get_height())
                        image.set_from_pixbuf(pix.scale_simple(max(1,int(pix.get_width()*ratio)), max(1,int(pix.get_height()*ratio)), GdkPixbuf.InterpType.BILINEAR))
                    except Exception as exc:
                        self.frame_info.set_text('图像解码失败：'+str(exc))
                        self.stop_play()
                        return
            sample = value.get('sample') or {}
            grip = (sample.get('gripper') or {}).get('position_mm')
            grip_text = '' if grip is None else f' · 夹爪 {grip:.1f} mm'
            missing = ' · 部分时刻缺少图像' if any(not f['png'] for f in value['frames']) else ''
            self.frame_info.set_text(f"有效片段 {value['segment']}/{value['segments']}{grip_text}{missing} · 回看已跳过暂停")
        self.submit(lambda: library.preview(request[1], request[2], request[3], request[4]), apply)

    def stop_play(self):
        if self.play_timer is not None:
            GLib.source_remove(self.play_timer)
            self.play_timer = None
        self.play.set_label('播放')

    def toggle_play(self, *_):
        if self.play_timer is not None:
            self.stop_play()
            return
        if not self.selected or not self.play.get_sensitive():
            return
        if self.slider.get_value() >= self.selected['duration']:
            self.slider.set_value(0)
        self.play.set_label('暂停')
        def tick():
            # Reading/decoding never queues unbounded work behind playback.
            if not self.preview_running:
                value = min(self.selected['duration'], self.slider.get_value()+.2)
                self.slider.set_value(value)
                if value >= self.selected['duration']:
                    self.play_timer = None
                    self.play.set_label('播放')
                    return False
            return True
        self.play_timer = GLib.timeout_add(200, tick)

    def update_action(self):
        state=self.recording.action_state()
        exporting=process_running(self.recording.export_process)
        can_export=self.root is not None and not self.busy and state.export
        self.recording.scan.set_sensitive(can_export)
        self.recording.export.set_sensitive(can_export)
        for widget in (self.recording.scan,self.recording.export):
            widget.set_tooltip_text('正在读取数据集' if self.busy else
                '请先选择有效数据集' if self.root is None else state.reason or '处理所选整个数据集')
        for widget in (self.datasets,self.filter,self.refresh_button):
            widget.set_sensitive(not self.busy and not exporting)
        valid = self.selected is not None and self.selected['category'] not in ('recording','invalid')
        self.move.set_label('恢复轨迹' if self.filter.get_active_id()=='trash' else '移入回收站')
        self.move.set_sensitive(valid and not self.busy and state.organize)
        can_play = valid and self.selected['duration']>0 and not self.busy
        self.play.set_sensitive(can_play)
        self.slider.set_sensitive(can_play)

    def move_selected(self, *_):
        if not self.move.get_sensitive() or not self.selected:
            return
        row, root = self.selected, self.root
        self.stop_play()
        self.desired_preview = None
        self.generation += 1
        self.set_busy(True)
        def apply(value, error):
            self.set_busy(False)
            self.notice.set_text(error or ('已恢复：' if row['trash'] else '已移入回收站，可切换到回收站恢复：')+row['id'])
            self.dataset_changed()
        self.submit(lambda: library.move_episode(root, row['id'], restore=row['trash']), apply)

    def close(self):
        self.closed = True
        self.stop_play()
        self.pool.shutdown(wait=False, cancel_futures=True)
