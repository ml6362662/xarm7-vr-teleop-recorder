"""Cleaning integration against isolated archives, including encoded outputs."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

import pytest

from export_recording import export_dataset
from recording_data import source_fingerprint
from recording_quality import scan_dataset
from test_recording import make_episode
from test_recording_gate import trigger_episode


@contextmanager
def edit_db(path):
    db=sqlite3.connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()


def change_sample(path, sequence, update):
    with edit_db(path/'raw.sqlite3') as db:
        value=json.loads(db.execute('SELECT payload FROM telemetry WHERE sequence=?',(sequence,)).fetchone()[0])
        update(value)
        db.execute('UPDATE telemetry SET payload=? WHERE sequence=?',(json.dumps(value),sequence))


def test_clean_export_keeps_good_episode_and_raw_bytes(tmp_path):
    import pyarrow.parquet as pq
    source=tmp_path/'raw'
    good=trigger_episode(source)
    bad=make_episode(source)
    # Index 1 is between export ticks: validation must not hide it at 15 FPS.
    change_sample(bad,1,lambda s:s.update(source_valid=False))
    before={p.name:source_fingerprint(p) for p in (good,bad)}
    report=scan_dataset(source,fps=15)
    assert report['summary']['accepted']==1 and report['summary']['rejected']==1
    out=export_dataset(source,tmp_path/'out',fps=15,clean=True)
    manifest=json.loads((out/'export_manifest.json').read_text())
    assert manifest['episodes'][0]['source']==str(good)
    assert manifest['cleaning']['rejected']==1
    table=pq.read_table(out/'data/chunk-000/episode_000000.parquet').to_pydict()
    assert sum(table['next.discontinuity'])==1
    assert sum(table['next.done'])==1 and table['next.success'][-1]
    assert set(table['capture.segment_index'])=={0,1}
    assert all(not 100.3 <= t < 100.7 for t in table['source.monotonic_s'])
    assert {p.name:source_fingerprint(p) for p in (good,bad)}==before


def test_failure_default_exclusion_explicit_inclusion_and_unknown(tmp_path):
    import pyarrow.parquet as pq
    path=make_episode(tmp_path/'raw',success=False)
    make_episode(tmp_path/'raw',success=None)
    report=scan_dataset(tmp_path/'raw')
    assert report['summary']==dict(total=2,accepted=0,rejected=0,excluded=2,warnings=0)
    out=export_dataset(tmp_path/'raw',tmp_path/'out',fps=15,include_failures=True,clean=True)
    report=json.loads((out/'quality_report.json').read_text())
    assert report['summary']['accepted']==1 and report['summary']['excluded']==1
    table=pq.read_table(out/'data/chunk-000/episode_000000.parquet').to_pydict()
    assert not any(table['next.success']) and table['annotation.success_valid'][-1]


@pytest.mark.parametrize('defect',['image_missing','png','dimension','sequence','trigger','time','quaternion'])
def test_defects_reported_without_touching_raw(tmp_path,defect):
    path=trigger_episode(tmp_path/'raw')
    if defect in ('image_missing','png','dimension'):
        with edit_db(path/'raw.sqlite3') as db:
            if defect=='image_missing':
                db.execute("DELETE FROM images WHERE camera='wrist' AND sequence=3")
            elif defect=='png':
                db.execute("UPDATE images SET png=x'00' WHERE camera='wrist' AND sequence=0")
            else:
                meta=json.loads((path/'episode.json').read_text())
                meta['cameras'][0]['width']=66
                (path/'episode.json').write_text(json.dumps(meta))
    elif defect=='sequence':
        with edit_db(path/'raw.sqlite3') as db:
            db.execute('DELETE FROM telemetry WHERE sequence=1')
    elif defect=='trigger':
        change_sample(path,1,lambda s:s['hand'].update(trigger_pressed=False))
    elif defect=='quaternion':
        change_sample(path,1,lambda s:s.update(tcp_quaternion_wxyz=[0,0,0,0]))
    else:
        change_sample(path,1,lambda s:s.update(monotonic_s=1))
    before=source_fingerprint(path)
    report=scan_dataset(tmp_path/'raw',action_space='tcp_pose' if defect=='quaternion' else 'joint')
    assert report['summary']['rejected']==1
    assert source_fingerprint(path)==before


def test_camera_15fps_must_not_be_upconverted_by_repeating_images(tmp_path):
    path=make_episode(tmp_path/'raw')
    with edit_db(path/'raw.sqlite3') as db:
        db.execute('DELETE FROM images WHERE sequence%2=1')
        # Frame numbering is consecutive in a camera that actually runs at 15 Hz.
        db.execute('UPDATE images SET sequence=-sequence-1')
        db.execute('UPDATE images SET sequence=(-sequence-1)/2')
    assert scan_dataset(tmp_path/'raw',fps=30)['summary']['rejected']==1
    assert scan_dataset(tmp_path/'raw',fps=15)['summary']['accepted']==1


def test_bad_last_interval_never_moves_success_to_earlier_clip(tmp_path):
    path=trigger_episode(tmp_path/'raw')
    meta=json.loads((path/'episode.json').read_text())
    meta['capture_intervals'][-1]['end_s']=100.72
    (path/'episode.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='连续片段'):
        export_dataset(tmp_path/'raw',tmp_path/'out',fps=15,clean=True)
    assert not (tmp_path/'out').exists()
    report=json.loads((tmp_path/'out.partial/quality_report.json').read_text())
    assert report['summary']['rejected']==1 and report['summary']['accepted']==0
    assert (tmp_path/'out.partial/EXPORT_FAILED.json').is_file()


def test_export_rechecks_data_after_scan(tmp_path):
    path=make_episode(tmp_path/'raw')
    assert scan_dataset(tmp_path/'raw')['summary']['accepted']==1
    change_sample(path,1,lambda s:s.update(source_valid=False))
    with pytest.raises(ValueError):
        export_dataset(tmp_path/'raw',tmp_path/'out',fps=15,clean=True)
    assert not (tmp_path/'out').exists()


def test_corrupt_metadata_does_not_abort_other_clean_episodes(tmp_path):
    make_episode(tmp_path/'raw')
    bad=make_episode(tmp_path/'raw')
    (bad/'episode.json').write_text('{broken')
    out=export_dataset(tmp_path/'raw',tmp_path/'out',fps=15,clean=True)
    report=json.loads((out/'quality_report.json').read_text())
    assert report['summary']['accepted']==1 and report['summary']['rejected']==1


def test_scan_never_reads_active_database(tmp_path):
    path=make_episode(tmp_path/'raw')
    (path/'raw.sqlite3-wal').touch()
    assert scan_dataset(tmp_path/'raw')['summary']['rejected']==1


def test_scan_reports_real_observation_delay(tmp_path):
    path=make_episode(tmp_path/'raw')
    change_sample(path,0,lambda s:s.update(observation_read_start_s=99.8,observation_read_end_s=99.81))
    report=scan_dataset(tmp_path/'raw')
    item=report['episodes'][0]
    assert item['max_observation_age_ms']==pytest.approx(190)
    assert any('25 ms' in warning for warning in item['warnings'])


@pytest.mark.parametrize('kind,fps',[('unknown',15),('lerobot21',0),('lerobot21',float('nan'))])
def test_invalid_export_options_create_no_output(tmp_path,kind,fps):
    make_episode(tmp_path/'raw')
    with pytest.raises(ValueError):
        export_dataset(tmp_path/'raw',tmp_path/'out',kind=kind,fps=fps)
    assert not (tmp_path/'out.partial').exists()


@pytest.mark.parametrize('scan_only',[True,False])
def test_cli_report_and_clean_export(tmp_path,scan_only):
    import subprocess
    import sys
    make_episode(tmp_path/'raw')
    bad=make_episode(tmp_path/'raw')
    change_sample(bad,1,lambda s:s.update(source_valid=False))
    script=Path(__file__).resolve().parents[1]/'export_recording.py'
    result=subprocess.run([sys.executable,str(script),str(tmp_path/'raw'),str(tmp_path/'out'),
                           '--fps','15','--scan-only' if scan_only else '--clean'],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    value=json.loads(result.stdout.splitlines()[-1])
    assert value['type']=='export_result' and value['summary']['rejected']==1
    assert value['summary']['accepted']==1
    assert (tmp_path/'out/quality_report.json').is_file()
    assert (tmp_path/'out/quality_report.txt').is_file()
    assert (tmp_path/'out/meta/info.json').exists() is not scan_only
