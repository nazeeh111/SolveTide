"""File publication retains no-clobber semantics across failed writes."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from solvetide import artifacts


def test_json_bytes_and_mode_match_existing_contract(tmp_path):
    out=tmp_path/'result.json'
    previous=os.umask(0o027)
    try: artifacts.save(out,{'x':[1,2],'label':'résumé'})
    finally: os.umask(previous)
    assert out.read_text()==json.dumps({'x':[1,2],'label':'résumé'},indent=2,allow_nan=False)+'\n'
    assert out.stat().st_mode & 0o777==0o640
    assert list(tmp_path.iterdir())==[out]


def test_failed_write_leaves_no_output_and_same_path_can_retry(tmp_path):
    out=tmp_path/'result.json'
    def fail(document,stream,**kwargs):
        stream.write('partial')
        raise OSError('synthetic write failure')
    with patch.object(artifacts.json,'dump',side_effect=fail):
        with pytest.raises(OSError,match='synthetic write failure'): artifacts.save(out,{'a':1})
    assert list(tmp_path.iterdir())==[]
    artifacts.save(out,{'a':1})
    assert json.loads(out.read_text())=={'a':1}


@pytest.mark.parametrize('kind',['file','symlink','directory'])
def test_existing_destination_never_replaced(tmp_path,kind):
    out=tmp_path/'result.json'
    if kind=='file':out.write_bytes(b'prior')
    elif kind=='symlink':out.symlink_to(tmp_path/'missing')
    else:out.mkdir()
    with pytest.raises(FileExistsError):artifacts.save(out,{'new':1})
    assert (out.read_bytes()==b'prior' if kind=='file' else out.is_symlink() if kind=='symlink' else out.is_dir())
    assert sorted(x.name for x in tmp_path.iterdir())==['result.json']


def test_stage_collision_does_not_remove_foreign_file(tmp_path):
    stage=tmp_path/'.solvetide-fixed.tmp';stage.write_bytes(b'foreign')
    with patch.object(artifacts.secrets,'token_hex',return_value='fixed'):
        with pytest.raises(FileExistsError):artifacts.save(tmp_path/'result.json',{'a':1})
    assert stage.read_bytes()==b'foreign'
    assert not (tmp_path/'result.json').exists()


def test_nonfinite_and_failed_fdopen_leave_no_stage(tmp_path):
    with pytest.raises(ValueError):artifacts.save(tmp_path/'result.json',{'bad':float('nan')})
    assert list(tmp_path.iterdir())==[]
    captured=[];real_open=os.open
    def open_capture(*args,**kwargs):
        fd=real_open(*args,**kwargs);captured.append(fd);return fd
    with patch.object(artifacts.os,'open',side_effect=open_capture),patch.object(artifacts.os,'fdopen',side_effect=OSError('fdopen failed')):
        with pytest.raises(OSError,match='fdopen failed'):artifacts.save(tmp_path/'result.json',{'a':1})
    with pytest.raises(OSError):os.fstat(captured[0])
    assert list(tmp_path.iterdir())==[]


def test_parallel_writers_have_one_complete_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    barrier=threading.Barrier(8);out=tmp_path/'result.json'
    def publish(index):
        barrier.wait()
        try: artifacts.save(out,{'writer':index,'payload':'x'*4096});return 'success'
        except FileExistsError:return 'exists'
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(publish,range(8)))
    assert results.count('success')==1 and results.count('exists')==7
    result=json.loads(out.read_text())
    assert result['writer'] in range(8) and result['payload']=='x'*4096
    assert list(tmp_path.iterdir())==[out]


def test_cleanup_failure_after_publish_reports_complete_output(tmp_path):
    out=tmp_path/'result.json';real_unlink=Path.unlink
    def fail_stage(path,*args,**kwargs):
        if path.name.startswith('.solvetide-'):raise OSError('synthetic cleanup failure')
        return real_unlink(path,*args,**kwargs)
    with patch.object(Path,'unlink',fail_stage),pytest.warns(RuntimeWarning,match='Result published'):
        artifacts.save(out,{'value':3})
    assert json.loads(out.read_text())=={'value':3}
    assert len(list(tmp_path.glob('.solvetide-*.tmp')))==1
