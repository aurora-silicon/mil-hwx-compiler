#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check diagnostic prefixes keep the source stream and reject corruption."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'research')]
from camera_prefixes import Graph, build as rebuild
from camera_exact_prefixes import build as exact
import inspect_anec as reader


def fixture(root):
    blob=bytearray(struct.pack('<II',1,2)+bytes(56))
    lines=['int32 axis = const()[val = int32(1)];',
           'bool interleave = const()[val = bool(false)];',
           'tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1,1])];',
           'tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1,1])];',
           'tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0,0,0,0])];',
           'int32 gp = const()[val = int32(1)];',
           'string pt = const()[val = string("valid")];']
    for i,(name,source,ci) in enumerate((('a','x',64),('b','a',64),('c','a',64),('y','d',128),('z','y',64),('end','z',64))):
        values=[((i+j)%13-6)/128 for j in range(64*ci)]
        payload=struct.pack('<'+'e'*len(values),*values)
        offset=len(blob)
        blob+=struct.pack('<IIQQ',0xdeadbeef,1,len(payload),offset+64)+bytes(40)+payload
        blob+=bytes(-len(blob)%64)
        if name=='y': lines.append('tensor<fp16, [1,128,8,8]> d = concat(values=(b,c),axis=axis,interleave=interleave);')
        lines.extend([f'tensor<fp16, [64,{ci},1,1]> w{i} = const()[val = tensor<fp16, [64,{ci},1,1]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset})))];',
                      f'tensor<fp16, [1,64,8,8]> {name} = conv(x={source},weight=w{i},strides=st,pad_type=pt,pad=pd,dilations=dl,groups=gp);'])
    text='program(1.3) {\nfunc main<ios18>(tensor<fp16, [1,64,8,8]> x) {\n'+'\n'.join(lines)+'\n} -> (end);\n}\n'
    (root/'model.mil').write_text(text);(root/'weights.bin').write_bytes(blob)
    return text


def main(compiler):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);text=fixture(root)
        g=Graph(text)
        prefix,boundary,count=g.source('b')
        assert boundary=='x' and count==2 and ' c = conv' not in prefix
        alone,boundary,count=g.source('b',isolated=True)
        assert boundary=='a' and count==1 and ' a = conv' not in alone
        try: Graph(text.replace('\nint32 axis','\nint32\naxis'))
        except ValueError: pass
        else: raise AssertionError('multiline input accepted silently')
        run=subprocess.run([compiler,'--target','H13','--schedule','linked','--mil',str(root/'model.mil'),
                            '--model-root',str(root),'--output',str(root/'source')],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        for name in ('model.mil','weights.bin'): (root/'source'/name).write_bytes((root/name).read_bytes())
        index=exact(root/'source',root/'exact')
        assert {r['result'] for r in index['records']}=={'a','d','y','z'},index
        original,_=reader.load_package(root/'source')
        orig=(root/'source/program-0.anec').read_bytes()
        old_constants=orig[4096+original['constantOffset']:]
        for record in index['records']:
            case=root/'exact'/record['directory'];m,_=reader.load_package(case)
            data=(case/'program-0.anec').read_bytes()
            assert data[4096+m['constantOffset']:]==old_constants
            bad=bytearray(data);bad[4096+2]^=1
            (case/'program-0.anec').write_bytes(bad)
            try: reader.load_package(case)
            except ValueError: pass
            else: raise AssertionError('prefix accepted wrong request NID')
            (case/'program-0.anec').write_bytes(data)
        l2run=subprocess.run([compiler,'--target','H13','--schedule','linked-l2','--mil',str(root/'model.mil'),
                              '--model-root',str(root),'--output',str(root/'resident')],capture_output=True,text=True)
        assert l2run.returncode==0,l2run.stderr
        l2,_=reader.load_package(root/'resident')
        assert l2['scratchBytes']<original['scratchBytes']
        assert l2['taskDescriptors']==original['taskDescriptors']
        assert l2['layers'][3]['zeroCopy'] is True
        assert l2['layers'][4]['outputEdge']=='l2-alias'
        assert l2['layers'][5]['inputEdge']=='l2-alias'
        assert l2['layers'][5]['outputEdge']=='l2-dependent'
        assert l2['layers'][6]['inputEdge']=='l2-dependent'
        assert 'y' not in l2['scratch']['regions'] and 'z' not in l2['scratch']['regions']
        from h13_l2_edge_parity import tasks
        dram_tasks=tasks(orig)
        l2bytes=(root/'resident/program-0.anec').read_bytes()
        l2tasks=tasks(l2bytes)
        assert old_constants==l2bytes[4096+l2['constantOffset']:]
        for (_,a),(_,b) in zip(dram_tasks,l2tasks):
            for address in a:
                if 0xc800<=address<0xd000 or 0x1f800<=address<0x1f900:
                    assert a[address]==b[address],(address,a[address],b[address])
        rebuilt=rebuild(root/'source',root/'rebuilt',compiler,['b','y'])
        assert len(rebuilt['records'])==4
        assert all(r['status']=='pass' for r in rebuilt['records']),rebuilt
    print('camera prefix closure, original-stream preservation, isolated layers and corruption rejection: PASS')


if __name__=='__main__': main(str(Path(sys.argv[1]).resolve()))
