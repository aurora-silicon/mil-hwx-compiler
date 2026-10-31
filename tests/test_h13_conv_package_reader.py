#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convolution package reader regression; generated weights, no device/fixtures."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research import inspect_anec


def source(cin,cout,k,stride,groups,relu):
    data=bytearray(struct.pack('<II',1,2)+bytes(56))
    offsets=[]
    for payload in (struct.pack('<e',.125)*(cout*(cin//groups)*k*k),bytes(cout*2)):
        offset=len(data);offsets.append(offset)
        data+=struct.pack('<IIQQ',0xDEADBEEF,1,len(payload),offset+64)+bytes(40)+payload
        data+=bytes(-len(data)%64)
    shape=[1,cin,12,10];out=[1,cout,(12+stride-1)//stride,(10+stride-1)//stride]
    p=k//2
    text=f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
 func main<ios18>(tensor<fp16, {shape}> x) {{
  string pt = const()[name = string("pt"), val = string("{'custom' if p else 'valid'}")];
  tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([{stride}, {stride}])];
  tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];
  tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];
  int32 gp = const()[name = string("gp"), val = int32({groups})];
  tensor<fp16, [{cout}, {cin//groups}, {k}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{cout}, {cin//groups}, {k}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[0]})))];
  tensor<fp16, [{cout}]> b = const()[name = string("b"), val = tensor<fp16, [{cout}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[1]})))];
  tensor<fp16, {out}> y = conv(x = x, weight = w, bias = b, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gp)[name = string("conv")];
'''
    if relu:text+=f'  tensor<fp16, {out}> r = relu(x = y)[name = string("relu")];\n'
    text+=' } -> ('+('r' if relu else 'y')+');\n}\n'
    return text,bytes(data)


def rejected(directory,program,tensors,change):
    changed=copy.deepcopy(program)
    change(changed)
    try:inspect_anec.validate_program(directory,changed,tensors)
    except ValueError:return
    raise AssertionError('malformed convolution package was accepted')


def main(compiler):
    with tempfile.TemporaryDirectory(prefix='h13-conv-reader-') as temporary:
        root=Path(temporary)
        for index,spec in enumerate(((16,32,3,1,1,False),(64,128,1,2,1,False),(64,64,3,1,64,True))):
            case=root/str(index);case.mkdir()
            mil,weights=source(*spec)
            (case/'model.mil').write_text(mil);(case/'weights.bin').write_bytes(weights)
            run=subprocess.run([str(compiler),'--target','H13','--mil',str(case/'model.mil'),
                '--model-root',str(case),'--output',str(case/'out')],capture_output=True,text=True,timeout=120)
            assert run.returncode==0,run.stderr
            manifest,_=inspect_anec.load_package(case/'out')
            program=manifest['programs'][0];tensors=manifest['tensors']
            assert program['operation']==('conv+relu' if spec[-1] else 'conv')
            rejected(case/'out',program,tensors,lambda p:p.update(encoder='unknown-conv'))
            rejected(case/'out',program,tensors,lambda p:p['inputs'][0].update(allocationBytes=p['inputs'][0]['allocationBytes']+1))
            rejected(case/'out',program,tensors,lambda p:p['outputs'][0]['nchw'].__setitem__(1,999))
            rejected(case/'out',program,tensors,lambda p:p['inputs'][0].update(dtype='bool'))
            if index<2:
                rejected(case/'out',program,tensors,lambda p:p.update(operation='add'))
    print('H13 convolution package reader: PASS (3 valid, 14 malformed)')


if __name__=='__main__':
    main(Path(sys.argv[1] if len(sys.argv)>1 else ROOT/'build/mil-hwxc').resolve())
