#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Expose DRAM intermediates without recompiling the preceding linked tasks.

All earlier task/register bytes, scratch addresses, constants and task order
are preserved. Only the selected layer's destination is rebound to output,
the terminal task is closed, and ANEC sizing/boundary metadata is adjusted.
Unfinished placement owners and placement-only results are skipped. This is
a diagnostic package, never a production camera qualification.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import inspect_anec as reader
from camera_prefixes import Graph, sha


def word_offsets(task):
    words = struct.unpack(f'<{len(task)//4}I',task)
    cursor = 10+(1 if words[9]&3 == 3 else 0)
    out = {}
    while cursor < len(words):
        count,base = (words[cursor]>>26)+1,words[cursor]&0x3ffffff
        for i in range(count): out[base+4*i] = (cursor+1+i)*4
        cursor += count+1
    return out


def make_prefix(directory, destination, layer_index):
    manifest,_ = reader.load_package(directory)
    original = manifest['programs'][0]
    if (original['encoder'] != 'h13-linked-native-graph' or
            any(l['inputEdge'] != 'dram' or l['outputEdge'] != 'dram' for l in original['layers'])):
        raise ValueError('exact prefixes require a DRAM linked native graph')
    layer = original['layers'][layer_index]
    result = layer['result']
    if (layer.get('surfaceOwner') != result or result not in manifest['scratch']['regions']):
        return None
    layers = copy.deepcopy(original['layers'][:layer_index+1])
    exposed = {l['result'] for l in layers if l.get('surfaceOwner') == result}
    regions = {n:copy.deepcopy(v) for n,v in manifest['scratch']['regions'].items()
               if n in {l['result'] for l in layers} and n not in exposed}
    if any(r.get('aliasOf',n) not in regions for n,r in regions.items()):
        return None
    data = (Path(directory)/original['file']).read_bytes()
    fields = list(reader.HEADER.unpack_from(data))
    tasks = reader.split_h13_tasks(data[4096:4096+fields[3]],fields[1]//4-1,fields[2])
    stop = layer['firstTask']+layer['taskDescriptors']
    starts = [0]
    for t in tasks[:-1]: starts.append(struct.unpack_from('<I',t,28)[0])
    text_size = starts[stop-1]+len(tasks[stop-1])
    constant_offset = (text_size+63)//64*64
    constants = data[4096+original['constantOffset']:]
    output_region = manifest['scratch']['regions'][result]
    shape = Graph((Path(directory)/'model.mil').read_text()).shapes[result]
    nchw = output_region['nchw']
    binding = dict(name=result,index=4,dtype='float16',shape=list(shape),
                   logicalBytes=math.prod(shape)*2,nchw=nchw,
                   allocationBytes=output_region['bytes'])
    text = bytearray(data[4096:4096+text_size])
    # Rebind every channel slice of the exposed layer, preserving its L2
    # compute/weight fields and the preceding prefix byte for byte.
    exposed_tasks = {i for l in layers if l['result'] in exposed
                     for i in range(l['firstTask'],l['firstTask']+l['taskDescriptors'])}
    for i in sorted(exposed_tasks):
        start = starts[i];positions = word_offsets(tasks[i])
        def get(address): return struct.unpack_from('<I',text,start+positions[address])[0]
        def put(address,value): struct.pack_into('<I',text,start+positions[address],value)
        if not get(0x17800)&1:
            raise ValueError('selected layer contains an internal non-DRAM write')
        selectors = struct.unpack_from('<I',text,start+32)[0]
        if (selectors>>12)&31 != 3: raise ValueError('selected producer does not write scratch')
        struct.pack_into('<I',text,start+32,(selectors&~(31<<12))|(4<<12))
        put(0x17804,get(0x17804)-output_region['offset'])
        put(0x17800,get(0x17800)&~0x300)
    end = starts[stop-1]
    header = list(struct.unpack_from('<10I',text,end))
    header[0] |= (1<<24)|(1<<25)
    header[1] &= 0xffff
    header[2] |= 0x22
    header[4] |= 2
    header[6] |= 0x9000
    header[7] = 0
    header[8] &= ~(1<<23)
    struct.pack_into('<10I',text,end,*header)
    fields[0] = constant_offset+len(constants)
    fields[2],fields[3] = stop,text_size
    fields[7] = (fields[0]+16383)//16384
    fields[7+4] = binding['allocationBytes']//16384
    fields[39+4*6:39+5*6] = nchw
    anec = (reader.HEADER.pack(*fields)+bytes(4096-reader.HEADER.size)+text+
            bytes(constant_offset-text_size)+constants)
    out = copy.deepcopy(manifest)
    p = out['programs'][0]
    for l in layers:
        if l['result'] in exposed: l['writeOffset'] -= output_region['offset']
    p.update(bytes=len(anec),constantOffset=constant_offset,taskDescriptors=stop,
             outputs=[binding],layers=layers)
    for key in reader.PROGRAM_FIELDS: out[key] = copy.deepcopy(p[key])
    out['layers'] = layers
    out['tasks'] = out['tasks'][:stop]
    out['scratch']['regions'] = regions
    out['tensors'] = {n:t for n,t in out['tensors'].items() if t['role'] != 'output'}
    out['tensors'][result] = dict(shape=list(shape),logicalBytes=math.prod(shape)*2,role='output')
    out['physicalOutputs'] = [dict(tensor=result,dtype='float16',shape=list(shape),logicalBytes=math.prod(shape)*2)]
    out['logicalResults'] = [dict(name=result,dtype='float16',shape=list(shape),conversion='identity',
                                physical=dict(tensor=result,elementOffset=0,elementCount=math.prod(shape)))]
    if layer_index == 0:
        # A one-layer prefix is an ordinary per-op program. Its compute
        # descriptor is still the original task, with the same NID/weights.
        p.update(operation=layer['operation'],encoder='h13-generic-conv')
        p.pop('layers')
        for key in ('schedule','scratch','tasks','layers'): out.pop(key,None)
        for key in reader.PROGRAM_FIELDS: out[key]=copy.deepcopy(p[key])
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/p['file']).write_bytes(anec)
    (destination/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    reader.load_package(destination)
    # Explicit proof for the promise in the module docstring.
    first_exposed = min(exposed_tasks)
    assert text[:starts[first_exposed]] == data[4096:4096+starts[first_exposed]]
    assert anec[4096+constant_offset:] == constants
    return dict(layer=layer_index,result=result,mode='exact-prefix',directory=destination.name,
                source_first_task=0,source_tasks=stop,checkpoint_source_tasks=sorted(exposed_tasks),
                tasks=stop,boundary=original['inputs'][0]['name'],shape=list(shape),status='pass',
                files={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})


def build(directory,output):
    directory,output = Path(directory),Path(output)
    manifest,_ = reader.load_package(directory)
    output.mkdir(parents=True,exist_ok=False)
    for source,name in ((directory/'LICENSE-MODEL','LICENSE-MODEL'),
                        (Path(__file__).resolve().parents[1]/'LICENSE','LICENSE-COMPILER')):
        if source.is_file(): (output/name).write_bytes(source.read_bytes())
    index = dict(schema='aurora.camera-prefixes.v1',diagnostic_only=True,
                 source_files={n:sha(directory/n) for n in ('model.mil','weights.bin','manifest.json','program-0.anec')},records=[])
    for i,layer in enumerate(manifest['programs'][0]['layers']):
        record = make_prefix(directory,output/f'{i:03d}-{layer["result"]}-exact',i)
        if record: index['records'].append(record)
    (output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return index


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('package',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();index=build(a.package,a.output)
    print(json.dumps(dict(prefixes=len(index['records'])),indent=2))
