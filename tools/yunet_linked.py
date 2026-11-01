#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lower the audited YuNet graph to conv/relu/nearest and placement views.

Max-pool is a nonnegative max tree built from exact gather/selection weights.
Image space-to-depth and score/head decoding remain at the host boundary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Lowering:
    def __init__(self):
        self.lines = []
        self.data = bytearray(struct.pack('<II', 1, 2)+bytes(56))
        self.index = 0
        self.values = {}
        self.shapes = {'x':[1, 16, 160, 160]}
        self.lines += ['int32 axis = const()[val = int32(1)];',
                       'bool interleave = const()[val = bool(false)];']

    def name(self):
        self.index += 1
        return 'v'+str(self.index)

    def tensor(self, value, shape, expression):
        self.shapes[value] = list(shape)
        self.lines.append(f'tensor<fp16, {list(shape)}> {value} = {expression};')
        return value

    def constant(self, array):
        array = np.ascontiguousarray(array, dtype='<f2')
        payload = array.tobytes()
        offset = len(self.data)
        self.data += struct.pack('<IIQQ', 0xDEADBEEF, 1, len(payload), offset+64)+bytes(40)+payload
        self.data += bytes(-len(self.data)%64)
        value = self.name()
        self.lines.append(f'tensor<fp16, {list(array.shape)}> {value} = const()[val = tensor<fp16, {list(array.shape)}>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset})))];')
        return value

    def conv(self, source, weight, bias=None, *, groups=1, stride=1, pads=(0,0,0,0), relu=False):
        w = self.constant(weight)
        b = self.constant(bias) if bias is not None else None
        _, _, height, width = self.shapes[source]
        co, _, kh, kw = weight.shape
        pt, pl, pb, pr = pads
        shape = [1, co, (height+pt+pb-kh)//stride+1, (width+pl+pr-kw)//stride+1]
        st, pd, dl, gp, mode = [self.name() for _ in range(5)]
        self.lines += [f'tensor<int32, [2]> {st} = const()[val = tensor<int32, [2]>([{stride},{stride}])];',
                       f'tensor<int32, [4]> {pd} = const()[val = tensor<int32, [4]>([{pt},{pb},{pl},{pr}])];',
                       f'tensor<int32, [2]> {dl} = const()[val = tensor<int32, [2]>([1,1])];',
                       f'int32 {gp} = const()[val = int32({groups})];',
                       f'string {mode} = const()[val = string("custom")];']
        result = self.tensor(self.name(), shape,
            f'conv(x={source},weight={w}'+(f',bias={b}' if b else '')+
            f',strides={st},pad_type={mode},pad={pd},dilations={dl},groups={gp})')
        return self.relu(result) if relu else result

    def relu(self, source):
        return self.tensor(self.name(), self.shapes[source], f'relu(x={source})')

    def concat(self, left, right):
        n,c,h,w = self.shapes[left]
        nr,cr,hr,wr = self.shapes[right]
        if (n,h,w)!=(nr,hr,wr):
            raise ValueError('concat geometry mismatch')
        return self.tensor(self.name(), [n,c+cr,h,w], f'concat(values=({left},{right}),axis=axis,interleave=interleave)')

    def nearest(self, source):
        n,c,height,width = self.shapes[source]
        sh,sw = self.name(),self.name()
        self.lines += [f'int32 {sh} = const()[val = int32(2)];', f'int32 {sw} = const()[val = int32(2)];']
        return self.tensor(self.name(), [n,c,height*2,width*2],
            f'upsample_nearest_neighbor(x={source},scale_factor_height={sh},scale_factor_width={sw})')

    def add(self, left, right):
        c = self.shapes[left][1]
        eye = np.eye(c, dtype=np.float16).reshape(c,c,1,1)
        # Clone both producers so ownership/adjacency can be proved locally.
        a,b = self.conv(left,eye),self.conv(right,eye)
        return self.conv(self.concat(a,b),np.concatenate((eye,eye),axis=1))

    def pool(self, source):
        c = self.shapes[source][1]
        phases=[]
        for block,(dy,dx) in enumerate(((1,1),(1,2),(2,1),(2,2))):
            gather = np.zeros((c,1,3,3),np.float16)
            gather[:,0,dy,dx] = 1
            phases.append(self.conv(source,gather,groups=c,pads=(1,1,1,1)))
        diff = np.zeros((4*c,4*c,1,1),np.float16)
        mix = np.zeros((2*c,4*c,1,1),np.float16)
        diff2 = np.zeros((2*c,2*c,1,1),np.float16)
        total = np.zeros((c,2*c,1,1),np.float16)
        for channel in range(c):
            for offset in (0,2*c):
                diff[offset+channel,offset+channel,0,0] = 1
                diff[offset+channel,offset+c+channel,0,0] = -1
                diff[offset+c+channel,offset+c+channel,0,0] = 1
            for pair in (0,1):
                mix[pair*c+channel,pair*2*c+channel,0,0] = 1
                mix[pair*c+channel,pair*2*c+c+channel,0,0] = 1
            diff2[channel,channel,0,0] = 1
            diff2[channel,c+channel,0,0] = -1
            diff2[c+channel,c+channel,0,0] = 1
            total[channel,channel,0,0] = total[channel,c+channel,0,0] = 1
        value=phases[0]
        for phase in phases[1:]: value=self.concat(value,phase)
        value = self.conv(value,diff,stride=2,relu=True)
        value = self.conv(value,mix)
        value = self.conv(value,diff2,relu=True)
        return self.conv(value,total)


def lower(model, output, compact_heads=True):
    model = model.resolve()
    metadata = json.loads((model/'model.json').read_text())
    if metadata.get('model')!='yunet' or metadata.get('input')!=[1,3,320,320]:
        raise ValueError('expected the audited YuNet 320 export')
    for name in ('model.onnx','model.mil','weights.bin'):
        if metadata['artifacts'].get(name)!=sha(model/name):
            raise ValueError('source export identity changed')
    network = onnx.shape_inference.infer_shapes(onnx.load(model/'model.onnx'))
    shapes = {v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*network.graph.input,*network.graph.value_info,*network.graph.output]}
    arrays = {v.name:nh.to_array(v) for v in network.graph.initializer}
    producer = {name:n for n in network.graph.node for name in n.output}
    builder = Lowering()
    builder.values[network.graph.input[0].name] = 'x'
    pools = 0
    for node in network.graph.node:
        attrs = {v.name:h.get_attribute_value(v) for v in node.attribute}
        kind = node.op_type
        if kind=='Constant':
            arrays[node.output[0]] = nh.to_array(attrs['value']);continue
        if kind=='Identity':
            if node.input[0] in arrays: arrays[node.output[0]] = arrays[node.input[0]]
            else: builder.values[node.output[0]] = builder.values[node.input[0]]
            continue
        if kind in ('Transpose','Reshape','Sigmoid'):
            continue # Checked and decoded only at returned heads below.
        x = builder.values[node.input[0]]
        if kind=='Conv':
            w = arrays[node.input[1]].astype(np.float16)
            b = arrays[node.input[2]].astype(np.float16) if len(node.input)>2 else None
            stride = attrs.get('strides',[1,1])
            if stride[0]!=stride[1] or attrs.get('dilations',[1,1])!=[1,1]:
                raise ValueError('unsupported convolution options')
            groups = attrs.get('group',1)
            if stride==[2,2]:
                if x!='x' or tuple(w.shape)!=(16,3,3,3) or attrs.get('pads')!=[1]*4:
                    raise ValueError('only the audited stem uses dense stride 2')
                stem = np.zeros((16,16,3,3),np.float16)
                for c in range(3):
                    for dy in range(3):
                        for dx in range(3):
                            qy,qx = dy-1,dx-1
                            stem[:,c*4+(qy%2)*2+qx%2,qy//2+1,qx//2+1] = w[:,c,dy,dx]
                y = builder.conv(x,stem,b,pads=(1,1,1,1))
            else:
                co,ci,kh,kw = w.shape
                physical_in = builder.shapes[x][1]
                # Four stride-8 heads each contribute an extra zero-padded
                # 64-channel block. This makes the twelve-head concat a
                # power-of-two input without four separate zero producers.
                small_head = 128 if compact_heads and shapes[node.output[0]][2:] == [40,40] else 64
                physical_out = max(small_head if co<16 else 16,1<<(co-1).bit_length())
                if groups==shapes[node.input[0]][1]==co and ci==1:
                    if physical_out!=physical_in: raise ValueError('depthwise padding mismatch')
                    groups = physical_in
                    padded = np.zeros((physical_out,1,kh,kw),np.float16);padded[:co] = w
                elif groups==1:
                    padded = np.zeros((physical_out,physical_in,kh,kw),np.float16)
                    padded[:co,:ci] = w
                else: raise ValueError('unsupported convolution grouping')
                bias = None
                if b is not None:
                    bias = np.zeros(physical_out,np.float16);bias[:co] = b
                y = builder.conv(x,padded,bias,groups=groups,pads=attrs.get('pads',[0]*4))
        elif kind=='Relu': y = builder.relu(x)
        elif kind=='Resize':
            if attrs.get('mode',b'nearest')!=b'nearest' or attrs.get('coordinate_transformation_mode')!=b'asymmetric':
                raise ValueError('unsupported resize convention')
            scales = arrays[node.input[2]] if len(node.input)>2 and node.input[2] else None
            if scales is None or list(scales)!=[1,1,2,2]: raise ValueError('resize must be exactly 2x')
            y = builder.nearest(x)
        elif kind=='Add': y = builder.add(x,builder.values[node.input[1]])
        elif kind=='MaxPool':
            if (attrs.get('kernel_shape')!=[2,2] or attrs.get('strides')!=[2,2] or
                any(attrs.get('pads',[0]*4)) or producer[node.input[0]].op_type!='Relu'):
                raise ValueError('pool rewrite requires nonnegative 2x2/s2 input')
            y = builder.pool(x);pools += 1
        else: raise ValueError('unsupported body operation: '+kind)
        builder.values[node.output[0]] = y
    heads, readback, head_offsets = [], [], []
    physical_channels=0
    if [v.name for v in network.graph.output]!=metadata['output_names']:
        raise ValueError('ONNX output order changed')
    decoded_channels=0
    for index,value in enumerate(network.graph.output):
        name = value.name; sigmoid = False
        while name not in builder.values:
            node = producer[name]
            if node.op_type not in ('Transpose','Reshape','Sigmoid'):
                raise ValueError('unsupported output decode')
            if node.op_type=='Sigmoid': sigmoid=True
            if node.op_type=='Transpose' and h.get_attribute_value(node.attribute[0])!=[0,2,3,1]:
                raise ValueError('output transpose changed')
            name = node.input[0]
        original = shapes[name]
        head = builder.values[name]
        if builder.shapes[head][1] not in (64,128) or original[2]!=original[3] or original[2] not in (10,20,40):
            raise ValueError('unexpected head geometry')
        while builder.shapes[head][2]<40: head=builder.nearest(head)
        head_offsets.append(physical_channels)
        physical_channels+=builder.shapes[head][1]
        heads.append(head)
        readback.append(dict(name=value.name,channel_offset=decoded_channels,channels=original[1],
                             spatial=original[2],sample_stride=40//original[2],sigmoid=sigmoid,
                             shape=metadata['output_shapes'][index]))
        decoded_channels+=original[1]
    # A power-of-two packed input lets one final 1x1 select the 48 useful
    # channels into a 64-channel readback, avoiding a multi-megabyte host copy.
    if not compact_heads:
        zero_source=next(value for value in builder.values.values()
                         if builder.shapes[value]==[1,64,40,40] and value not in heads)
        for _ in range(4):
            heads.append(builder.conv(zero_source,np.zeros((64,64,1,1),np.float16)))
            physical_channels+=64
    if physical_channels!=1024:
        raise ValueError('head concat must supply a power-of-two 1024-channel input')
    packed = heads[0]
    for head in heads[1:]: packed=builder.concat(packed,head)
    select=np.zeros((64,1024,1,1),np.float16)
    for i,item in enumerate(readback):
        for c in range(item['channels']): select[item['channel_offset']+c,head_offsets[i]+c,0,0]=1
    packed=builder.conv(packed,select)
    if pools!=4 or len(readback)!=12 or decoded_channels!=48: raise ValueError('expected four pools and twelve heads')
    output.mkdir(parents=True,exist_ok=False)
    mil = ('program(1.3) {\nfunc main<ios18>(tensor<fp16, [1,16,160,160]> x) {\n'+
           '\n'.join(builder.lines)+'\n} -> ('+packed+');\n}\n')
    (output/'model.mil').write_text(mil)
    (output/'weights.bin').write_bytes(builder.data)
    contract = dict(schema='aurora.yunet-conv-lowering.v1',model='yunet',
                    source_sha256=metadata['source_sha256'],onnx_sha256=sha(model/'model.onnx'),
                    input_shape=metadata['input'],output_shapes=metadata['output_shapes'],
                    output_names=metadata['output_names'],mil_outputs=metadata['mil_outputs'],
                    input_transform='yunet-space-to-depth-2',program_input_shape=[1,16,160,160],
                    program_outputs=[packed],program_output_shapes=[builder.shapes[packed]],readback=readback)
    (output/'lowering.json').write_text(json.dumps(contract,indent=2)+'\n')
    return contract


def compile_package(model, output, compiler, contract, schedule='linked'):
    compiler=compiler.resolve()
    package=output/'package'
    run=subprocess.run([str(compiler),'--target','H13','--schedule',schedule,
        '--mil',str(output/'model.mil'),'--model-root',str(output),'--output',str(package)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
    (output/'compiler.log').write_bytes(run.stdout)
    if run.returncode:
        raise ValueError(f'linked compiler refused the lowered graph (rc {run.returncode})')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
    import inspect_anec
    manifest,_=inspect_anec.load_package(package)
    if len(manifest['programs'])!=1 or manifest['dispatchPlan']!=[0]:
        raise ValueError('lowered YuNet is not one dispatched program')
    program=manifest['programs'][0]
    if ([v['shape'] for v in program['inputs']]!=[contract['program_input_shape']] or
        [v['name'] for v in program['outputs']]!=contract['program_outputs'] or
        [v['shape'] for v in program['outputs']]!=contract['program_output_shapes']):
        raise ValueError('lowered program boundary mismatch')
    for name in ('model.mil','weights.bin'):
        (package/name).write_bytes((output/name).read_bytes())
    licence=model.parent/'YUNET-LICENSE'
    (package/'LICENSE-MODEL').write_bytes(licence.read_bytes())
    plan=dict(contract)
    plan.update(schema='aurora.camera-linked-lowered.v1',target='H13',schedule=schedule,
        compiler_sha256=sha(compiler),files={name:sha(package/name) for name in
        ('manifest.json',program['file'],'model.mil','weights.bin','LICENSE-MODEL')})
    (package/'camera-linked.json').write_text(json.dumps(plan,indent=2)+'\n')
    return dict(status='pass',programs=1,dispatches=1,tasks=program['taskDescriptors'],
                scratch_bytes=program['scratchBytes'],package_sha256=sha(package/'camera-linked.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',type=Path)
    p.add_argument('--schedule',choices=('linked','linked-l2'),default='linked')
    p.add_argument('--head-padding',choices=('compact','legacy'),default='compact')
    a=p.parse_args()
    try:
        contract=lower(a.model,a.output,a.head_padding=='compact')
        result=compile_package(a.model,a.output,a.compiler,contract,a.schedule) if a.compiler else contract
    except Exception as error:
        if a.output.is_dir(): (a.output/'build-failure.json').write_text(json.dumps(dict(status='fail',error=str(error)),indent=2)+'\n')
        raise
    print(json.dumps(result,indent=2))
