"""Versioned solve evidence, exported only after accepted forward and adjoint."""
from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import warnings
from .checker import digest
from . import __version__


def matrix(a):
    a=a.tocsr(copy=True);a.sum_duplicates();a.sort_indices();a.eliminate_zeros()
    return {'shape':list(a.shape),'indptr':a.indptr.tolist(),'indices':a.indices.tolist(),'data':a.data.tolist()}


def bundle(system,forward,gradient,config,target=None):
    if not forward.accepted or not gradient.accepted:raise ValueError('accepted forward AND adjoint required')
    _,dj=system.objective(forward.x,target)
    doc={'schema_version':1,'artifact':'solvetide-discrete-result','package_version':__version__,
         'input':{'family':system.family,'n':system.n,'boundary':system.boundary,'parameters':dict(system.parameters),'target':target},
         'identity':dict(forward.identity),'assembly_identity':dict(system.identity),
         'acceptance':{'rtol':config.rtol,'atol':config.atol},'objective':gradient.objective,'gradient':gradient.gradient,
         'forward':forward.summary(),'adjoint':gradient.adjoint.summary(),
         'mesh':{'coordinates':system.coordinates.tolist(),'cells':system.cells.tolist(),'free':system.free.tolist(),'fixed':system.fixed.tolist(),'prescribed':system.prescribed.tolist()},
         'field':system.expand(forward.x).tolist(),
         'equations':{'A':matrix(system.A),'b':system.b.tolist(),'x':forward.x.tolist(),
                      'dJdx':dj.tolist(),'adjoint':gradient.adjoint.x.tolist(),
                      'derivatives':{k:{'A':matrix(da),'b':db.tolist()} for k,(da,db) in system.derivatives.items()}},
         'limits':'Normwise discrete numerical checks; no exact enclosure or physical-model validation. Memory is process-local; restart begins empty.'}
    doc['content_sha256']=digest(doc)
    return doc


def save(path,document):
    path=Path(path)
    stage=path.parent / f'.solvetide-{secrets.token_hex(16)}.tmp'
    fd=os.open(stage,os.O_WRONLY | os.O_CREAT | os.O_EXCL,0o666)
    published=False
    try:
        try:
            stream=os.fdopen(fd,'w',encoding='utf-8')
        except BaseException:
            os.close(fd)
            raise
        with stream:
            json.dump(document,stream,indent=2,allow_nan=False);stream.write('\n')
        os.link(stage,path)
        published=True
    finally:
        try:
            stage.unlink()
        except OSError as error:
            if not published:
                raise OSError(f'Could not remove incomplete staging file {stage}: {error}') from error
            with warnings.catch_warnings():
                warnings.simplefilter('always',RuntimeWarning)
                warnings.warn(f'Result published at {path}; staging file remains at {stage}: {error}',RuntimeWarning)
