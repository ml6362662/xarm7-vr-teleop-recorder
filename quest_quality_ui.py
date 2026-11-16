"""Compact quality browser with evidence details and raw-episode navigation."""
import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk, Pango
from quest_quality_model import STATES, counts, visible_episodes, severity, reasons, details


def show_report(recording,report):
    if recording.report_dialog is not None:
        recording.report_dialog.present()
        return
    dialog=Gtk.Dialog(title='轨迹质量 / 异常检查',transient_for=recording.owner,modal=False)
    recording.report_dialog=dialog
    dialog.set_default_size(1000,650)
    dialog.add_button('关闭',Gtk.ResponseType.CLOSE)
    def close(*_):
        recording.report_dialog=None
        dialog.destroy()
    dialog.connect('response',close)
    box=dialog.get_content_area();box.set_spacing(8);box.set_border_width(12)
    count=counts(report)
    summary=Gtk.Box(spacing=18)
    for key in ('rejected','warning','accepted','excluded','unknown'):
        if key=='unknown' and not count[key]:continue
        text,color=STATES[key]
        label=Gtk.Label()
        label.set_markup(f'<span foreground="{color}" size="large"><b>{text} {count[key]}</b></span>')
        summary.pack_start(label,False,False,0)
    box.pack_start(summary,False,False,0)
    policy=Gtk.Label(label='黄色警告不自动排除；灰色为筛选结果，不代表损坏。原始数据保持不变。',xalign=0)
    policy.set_line_wrap(True)
    box.pack_start(policy,False,False,0)
    source=Gtk.Label(label=f"{report['source']} · {report.get('fps','—')} FPS · 本次报告快照，数据变更后请重查",xalign=0)
    source.set_line_wrap(True);source.set_selectable(True)
    box.pack_start(source,False,False,0)
    bar=Gtk.Box(spacing=8)
    filtering=Gtk.ComboBoxText()
    for key,label in [('problems','仅问题 / 需复核'),('rejected','不合格'),('warning','需复核'),('excluded','未纳入'),('all','全部')]:filtering.append(key,label)
    filtering.set_active_id('problems')
    shown=Gtk.Label(xalign=0)
    replay=Gtk.Button(label='回看选中轨迹');replay.set_sensitive(False)
    for widget in (filtering,shown,replay):bar.pack_start(widget,False,False,0)
    box.pack_start(bar,False,False,0)
    pane=Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL);pane.set_position(480)
    box.pack_start(pane,True,True,0)
    store=Gtk.ListStore(str,str,str,str)
    tree=Gtk.TreeView(model=store)
    for i,title in enumerate(('轨迹 ID','质量','问题摘要')):
        cell=Gtk.CellRendererText()
        cell.set_property('ellipsize',Pango.EllipsizeMode.END)
        column=Gtk.TreeViewColumn(title,cell,text=i,foreground=3)
        column.set_resizable(True);column.set_max_width(230)
        tree.append_column(column)
    scroll=Gtk.ScrolledWindow();scroll.add(tree);pane.pack1(scroll,True,False)
    detail=Gtk.Label(label='选择轨迹查看检查依据和处理建议。',xalign=0,yalign=0)
    detail.set_line_wrap(True);detail.set_max_width_chars(48);detail.set_selectable(True)
    right=Gtk.ScrolledWindow();right.add(detail);pane.pack2(right,True,False)
    indexed={item['episode_id']:item for item in report['episodes']}
    selected=[None]
    def select(selection):
        model,iterator=selection.get_selected()
        selected[0]=None if iterator is None else indexed[model[iterator][0]]
        replay.set_sensitive(selected[0] is not None)
        detail.set_text(details(selected[0]) if selected[0] else '当前筛选下没有轨迹。可切换“全部”查看通过和未纳入项。')
    tree.get_selection().connect('changed',select)
    def refresh(*_):
        store.clear()
        items=visible_episodes(report,filtering.get_active_id())
        for item in items:
            label,color=STATES[severity(item)]
            store.append([item['episode_id'],label,reasons(item).replace('\n','；'),color])
        shown.set_text(f'显示 {len(items)} / {len(indexed)} 条')
        if items:tree.get_selection().select_path(Gtk.TreePath.new_first())
        else:select(tree.get_selection())
    def navigate(*_):
        if selected[0] is None:return
        ok,message=recording.library.reveal_episode(report['source'],selected[0]['episode_id'])
        if ok:dialog.response(Gtk.ResponseType.CLOSE)
        else:detail.set_text(details(selected[0])+'\n\n无法回看：'+message)
    replay.connect('clicked',navigate)
    tree.connect('row-activated',navigate)
    filtering.connect('changed',refresh)
    refresh();dialog.show_all()
