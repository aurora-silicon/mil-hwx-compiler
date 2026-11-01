#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Placement-only concat is accepted only when its complete tree aliases."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import inspect_anec


def main():
    compiler=str(Path(sys.argv[1] if len(sys.argv)>1 else 'build/mil-hwxc').resolve())
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        # Only procedural identity coefficients; no private or model data.
        payload=b''.join(struct.pack('<e',float(i==j)) for i in range(64) for j in range(64))
        (root/'weights.bin').write_bytes(struct.pack('<II',1,2)+bytes(56)+
            struct.pack('<IIQQ',0xDEADBEEF,1,len(payload),128)+bytes(40)+payload)
        prefix='''program(1.3) { func main<ios18>(tensor<fp16, [1,64,40,40]> x) {
        string pt=const()[val=string("valid")];
        tensor<int32,[2]> st=const()[val=tensor<int32,[2]>([1,1])];
        tensor<int32,[4]> pd=const()[val=tensor<int32,[4]>([0,0,0,0])];
        int32 gp=const()[val=int32(1)]; int32 axis=const()[val=int32(1)];
        tensor<fp16,[64,64,1,1]> w=const()[val=tensor<fp16,[64,64,1,1]>(BLOBFILE(path=string("@model_path/weights.bin"),offset=uint64(64)))];
        '''
        conv=lambda result,source:f'tensor<fp16,[1,64,40,40]> {result}=conv(x={source},weight=w,strides=st,pad_type=pt,pad=pd,dilations=st,groups=gp);\n'
        body=conv('a','x')+conv('b','x')+'''tensor<fp16,[1,128,40,40]> ab=concat(values=(a,b),axis=axis);
        '''+conv('c','x')+'''tensor<fp16,[1,192,40,40]> y=concat(values=(ab,c),axis=axis);
        } -> (y); }'''
        (root/'model.mil').write_text(prefix+body)
        run=subprocess.run([compiler,'--target','H13','--schedule','linked','--mil',str(root/'model.mil'),
            '--model-root',temp,'--output',str(root/'out')],capture_output=True,text=True,timeout=120)
        assert run.returncode==0,run.stderr
        m,_=inspect_anec.load_package(root/'out');p=m['programs'][0]
        assert p['taskDescriptors']==3 and m['scratch']['bytes']==0
        assert [v['taskDescriptors'] for v in p['layers']]==[1,1,0,1,0]
        assert [v['zeroCopy'] for v in p['layers']]==[False,False,True,False,True]
        # An extra use of a means the first concat cannot own both producers.
        bad=body.replace('values=(ab,c)','values=(ab,c)').replace(conv('c','x'),conv('c','a'))
        (root/'bad.mil').write_text(prefix+bad)
        run=subprocess.run([compiler,'--target','H13','--schedule','linked','--mil',str(root/'bad.mil'),
            '--model-root',temp,'--output',str(root/'refused')],capture_output=True,text=True,timeout=120)
        assert run.returncode and 'placement-only concat' in run.stderr,run.stderr
    print('H13 placement tree: PASS (three writers, zero concat tasks, shared producer refused)')


if __name__=='__main__': main()
