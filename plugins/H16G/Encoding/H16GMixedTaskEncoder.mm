#import "H16GMixedTaskEncoder.h"

#import "H16GDecodedFormValidator.h"
#import "H16GLUTEncoder.h"
#import "H16GTDWriter.h"

static BOOL put(H16GTDWriter *writer, NSUInteger base, NSUInteger offset,
                uint32_t value, NSString *field, NSError **error) {
    return [writer writeUInt32:value atOffset:base + offset
                         field:field error:error];
}

#define FIELD(offset, value, name) do { \
    if (!put(writer, base, offset, value, name, error)) return NO; \
} while (0)

static uint32_t taskStamp(ANEScheduledTask *task, uint32_t lowBits) {
    return (uint32_t)(task.index << 16) | lowBits;
}

static BOOL taskHasCommand(ANEScheduledTask *task,
                           ANEScheduledCommandKind kind) {
    for (ANEScheduledCommand *command in task.commands)
        if (command.kind == kind) return YES;
    return NO;
}

typedef NS_ENUM(NSUInteger,H16GDecodedTaskForm) {
    H16GDecodedTaskFormUnknown,
    H16GDecodedTaskFormSplit,
    H16GDecodedTaskFormTransposePair,
    H16GDecodedTaskFormMatmul,
    H16GDecodedTaskFormScale,
    H16GDecodedTaskFormReduceMax,
    H16GDecodedTaskFormCenterExp,
    H16GDecodedTaskFormReduceSum,
    H16GDecodedTaskFormReciprocal,
    H16GDecodedTaskFormNormalizeTranspose,
};

static BOOL taskOperationsAre(ANEOperationGraph *graph,ANEScheduledTask *task,
                              NSArray<NSString *> *names) {
    if(task.sourceNodeIdentifiers.count!=names.count) return NO;
    for(NSUInteger i=0;i<names.count;++i) {
        ANEOperationNode *node=[graph nodeForValueName:
            task.sourceNodeIdentifiers[i]];
        if(![node.operationName isEqualToString:names[i]]) return NO;
    }
    return YES;
}

static H16GDecodedTaskForm decodedForm(ANEOperationGraph *graph,
                                      ANEScheduledTask *task) {
    if(taskOperationsAre(graph,task,@[@"slice_by_size",@"slice_by_size",
                                      @"slice_by_size"]))
        return H16GDecodedTaskFormSplit;
    if(taskOperationsAre(graph,task,@[@"transpose",@"transpose"]))
        return H16GDecodedTaskFormTransposePair;
    if(taskOperationsAre(graph,task,@[@"matmul"]) ||
       taskOperationsAre(graph,task,@[@"matmul",@"transpose",@"reshape"]))
        return H16GDecodedTaskFormMatmul;
    if(taskOperationsAre(graph,task,@[@"mul"]))
        return H16GDecodedTaskFormScale;
    if(taskOperationsAre(graph,task,@[@"reduce_max"]))
        return H16GDecodedTaskFormReduceMax;
    if(taskOperationsAre(graph,task,@[@"sub",@"exp"]))
        return H16GDecodedTaskFormCenterExp;
    if(taskOperationsAre(graph,task,@[@"reduce_sum"]))
        return H16GDecodedTaskFormReduceSum;
    if(taskOperationsAre(graph,task,@[@"reciprocal"]))
        return H16GDecodedTaskFormReciprocal;
    if(taskOperationsAre(graph,task,@[@"mul",@"transpose"]))
        return H16GDecodedTaskFormNormalizeTranspose;
    return H16GDecodedTaskFormUnknown;
}

static NSUInteger encodedSize(H16GDecodedTaskForm form,ANEScheduledTask *task) {
    switch(form) {
        case H16GDecodedTaskFormSplit: return 0xe0;
        case H16GDecodedTaskFormTransposePair: return 0xe0;
        case H16GDecodedTaskFormMatmul:
            return taskHasCommand(task,ANEScheduledCommandKindDMAStore)
                ? 0x1e8 : 0x1f0;
        case H16GDecodedTaskFormScale: return 0x90;
        case H16GDecodedTaskFormReduceMax: return 0x90;
        case H16GDecodedTaskFormCenterExp: return 0xf0;
        case H16GDecodedTaskFormReduceSum: return 0x80;
        case H16GDecodedTaskFormReciprocal: return 0x90;
        case H16GDecodedTaskFormNormalizeTranspose: return 0x90;
        case H16GDecodedTaskFormUnknown: return 0;
    }
}

static BOOL writeMatrixAddressTable(H16GTDWriter *writer, NSUInteger base,
                                    uint32_t stride, NSError **error) {
    for (NSUInteger lane = 0; lane < 15; ++lane)
        FIELD(0x03c + lane * 4, (uint32_t)(lane + 1) * stride,
              @"matmul.tile_address");
    for (NSUInteger lane = 0; lane < 16; ++lane)
        FIELD(0x078 + lane * 4, 0x200, @"matmul.tile_extent");
    return YES;
}

static BOOL writeTaskCompletionTable(H16GTDWriter *writer, NSUInteger base,
                                     NSUInteger firstOffset,
                                     ANEScheduledTask *task,
                                     NSError **error) {
    for (NSUInteger lane = 0; lane < 16; ++lane) {
        FIELD(firstOffset + lane * 8, 0x22001548u + (uint32_t)lane,
              @"task.completion_register");
        FIELD(firstOffset + 4 + lane * 8, taskStamp(task, 0x41),
              @"task.completion_stamp");
    }
    return YES;
}

static BOOL writeSplitTask(H16GTDWriter *writer, NSUInteger base,
                           ANEScheduledTask *task, ANEOperationGraph *graph,
                           NSError **error) {
    ANEOperationNode *slice=[graph nodeForValueName:
        task.sourceNodeIdentifiers.firstObject];
    uint32_t rows=slice.outputType.shape.lastObject.unsignedIntValue;
    FIELD(0x000, 1, @"split.program_start");
    FIELD(0x010, 0x00340000, @"split.packet_words");
    FIELD(0x018, 0x68, @"split.record_count");
    FIELD(0x020, 0x00fff868, @"split.dispatch_limit");
    FIELD(0x030, 1, @"split.source_count");
    FIELD(0x034, 0x00010001, @"split.tensor_rank");
    FIELD(0x038, rows, @"split.rows");
    FIELD(0x03c, 1, @"split.batch");
    FIELD(0x040, rows, @"split.columns");
    FIELD(0x044, 0x95458005, @"split.layout_mode");
    FIELD(0x048, rows, @"split.output_rows");
    FIELD(0x04c, 1, @"split.output_batch");
    FIELD(0x050, rows, @"split.output_columns");
    FIELD(0x054, 4, @"split.head_count");
    FIELD(0x058, 1, @"split.element_bytes_log2");
    FIELD(0x05c, 0x20, @"split.vector_width");
    FIELD(0x060, 6, @"split.slice_count_code");
    FIELD(0x068, 0x80281342, @"split.dma_load");
    FIELD(0x06c, 0xce, @"split.dma_load_flags");
    FIELD(0x070, 0x600, @"split.output_span");
    FIELD(0x074, 0x180, @"split.input_row_bytes");
    FIELD(0x078, 0x135a, @"split.input_geometry");
    FIELD(0x07c, 0x01002031, @"split.input_descriptor");
    FIELD(0x080, 0x9041, @"split.input_stride_mode");
    FIELD(0x084, 0x00500172, @"split.input_stride");
    FIELD(0x088, 0x00500130, @"split.output_stride");
    FIELD(0x08c, 0x98039045, @"split.layout_command");
    FIELD(0x090, 0x80, @"split.row_bytes");
    FIELD(0x094, 0x2000, @"split.q_bytes");
    FIELD(0x098, 0x2000, @"split.k_bytes");
    FIELD(0x09c, 0x2000, @"split.v_bytes");
    FIELD(0x0a0, 0x0050047a, @"split.output_geometry");
    FIELD(0x0a4, 0x2000, @"split.output_plane_bytes");
    FIELD(0x0a8, 0x80811445, @"split.sram_store");
    FIELD(0x0ac, 0x200, @"split.sram_row_stride");
    FIELD(0x0b0, 0x2000, @"split.sram_plane_stride");
    FIELD(0x0b4, 0x04302031, @"split.sram_descriptor");
    FIELD(0x0b8, 0x23009344, @"split.wait");
    FIELD(0x0bc, 0x100, @"split.wait_mask");
    FIELD(0x0c4, 0x21809442, @"split.signal");
    FIELD(0x0d0, 0x22001340, @"split.task_begin");
    FIELD(0x0d4, taskStamp(task, 0x21), @"split.task_stamp");
    FIELD(0x0d8, 0x22001440, @"split.task_end");
    FIELD(0x0dc, 0x010000f1, @"split.task_end_stamp");
    return YES;
}

static BOOL writeTransposeTask(H16GTDWriter *writer, NSUInteger base,
                               ANEScheduledTask *task, NSError **error) {
    uint32_t rows = (uint32_t)task.tilePlan.rows;
    FIELD(0x000, (0x36u << 16) | (uint32_t)task.index, @"transpose.header");
    FIELD(0x010, 0x20, @"transpose.dependency_mask");
    FIELD(0x020, 0x00050001, @"transpose.control");
    FIELD(0x024, 0x00010001, @"transpose.tensor_rank");
    FIELD(0x028, rows, @"transpose.rows");
    FIELD(0x02c, 4, @"transpose.head_count");
    FIELD(0x030, rows, @"transpose.columns");
    FIELD(0x034, 0x93618005, @"transpose.layout_mode");
    FIELD(0x038, rows, @"transpose.output_rows");
    FIELD(0x03c, 4, @"transpose.output_heads");
    FIELD(0x040, rows, @"transpose.output_columns");
    FIELD(0x044, 0x00044000, @"transpose.permutation");
    FIELD(0x048, 4, @"transpose.rank");
    FIELD(0x04c, 0x30222200, @"transpose.axis_strides");
    FIELD(0x050, 2, @"transpose.element_bytes");
    FIELD(0x054, 0x00100000, @"transpose.surface_limit");
    FIELD(0x058, 0x800c1342, @"transpose.dma_load");
    FIELD(0x05c, 0xce, @"transpose.dma_load_flags");
    FIELD(0x060, 0x180, @"transpose.input_row_bytes");
    FIELD(0x064, 0x600, @"transpose.input_span");
    FIELD(0x068, 0x135a, @"transpose.input_geometry");
    FIELD(0x06c, 0x01002031, @"transpose.input_descriptor");
    FIELD(0x070, 0x803c1041, @"transpose.read_command");
    FIELD(0x074, 0x00400142, @"transpose.read_geometry");
    FIELD(0x078, rows, @"transpose.read_rows");
    FIELD(0x07c, 0x1050, @"transpose.read_stride");
    FIELD(0x080, 0x1000, @"transpose.q_offset");
    FIELD(0x084, 0x1000, @"transpose.k_offset");
    FIELD(0x088, 0x9052, @"transpose.write_command");
    FIELD(0x08c, 0x44a, @"transpose.write_geometry");
    FIELD(0x090, 0x4140, @"transpose.write_stride");
    FIELD(0x094, 0x80021241, @"transpose.dma_inter");
    FIELD(0x098, 0x00103c0c, @"transpose.dtype");
    FIELD(0x09c, 0x3c00, @"transpose.scale");
    FIELD(0x0a0, 0x81009444, @"transpose.sram_store");
    FIELD(0x0a4, 0x200, @"transpose.sram_row_stride");
    FIELD(0x0a8, 0x800, @"transpose.sram_head_stride");
    FIELD(0x0ac, 0x04302031, @"transpose.sram_descriptor");
    FIELD(0x0b0, 0x23009344, @"transpose.wait");
    FIELD(0x0bc, 0x21809442, @"transpose.signal");
    FIELD(0x0c0, 0x8000, @"transpose.signal_mask");
    FIELD(0x0c8, 0x22001340, @"transpose.task_begin");
    FIELD(0x0cc, taskStamp(task, 0x21), @"transpose.task_stamp");
    FIELD(0x0d0, 0x22001440, @"transpose.task_end");
    FIELD(0x0d4, 0x010000f1 | (uint32_t)(task.index << 16),
          @"transpose.task_end_stamp");
    return YES;
}

static BOOL writeMatmulTask(H16GTDWriter *writer, NSUInteger base,
                            ANEScheduledTask *task, NSError **error) {
    BOOL storesOutput=taskHasCommand(task,ANEScheduledCommandKindDMAStore);
    uint32_t rows = (uint32_t)task.tilePlan.rows;
    uint32_t packetWords = storesOutput ? 0x7a : 0x79;
    FIELD(0x000, (packetWords << 16) | (uint32_t)task.index, @"matmul.header");
    if (storesOutput) FIELD(0x008, 0x04000000, @"matmul.output_flag");
    FIELD(0x010, storesOutput ? 0x00fff860 : 0x00fff820,
          @"matmul.dispatch_limit");
    if (!storesOutput) FIELD(0x01c, 3, @"matmul.dependency_mode");
    FIELD(0x020, storesOutput ? 0x0005000b : 0x00050003,
          @"matmul.control");
    if (!storesOutput) FIELD(0x024, 1, @"matmul.input_pair_count");
    FIELD(0x028, 0x80301540, @"matmul.tile_control");
    FIELD(0x02c, storesOutput ? 0x000a02c0 : 0x00030240,
          @"matmul.task_control");
    FIELD(0x030, storesOutput ? 0x2000 : 0x200, @"matmul.output_tile_bytes");
    FIELD(0x034, storesOutput ? 0x2000 : 0x8000, @"matmul.input_tile_bytes");
    FIELD(0x038, 0x000f1559, @"matmul.tile_table_header");
    if (!writeMatrixAddressTable(writer,base,storesOutput?0x200:0x800,error))
        return NO;
    FIELD(0x0b8, 0x00010001, @"matmul.tensor_rank");
    FIELD(0x0bc, rows, @"matmul.rows");
    FIELD(0x0c0, 1, @"matmul.batch");
    FIELD(0x0c4, rows, @"matmul.columns");
    FIELD(0x0c8, 0x93458005, @"matmul.compute_mode");
    FIELD(0x0cc, rows, @"matmul.output_rows");
    FIELD(0x0d0, 1, @"matmul.output_batch");
    FIELD(0x0d4, rows, @"matmul.output_columns");
    FIELD(0x0d8, 4, @"matmul.head_count");
    FIELD(0x0dc, 1, @"matmul.element_bytes_log2");
    FIELD(0x0e0, 0x20200004, @"matmul.accumulator_geometry");
    FIELD(0x0e4, 2, @"matmul.element_bytes");
    FIELD(0x0e8, 0x00100000, @"matmul.surface_limit");
    FIELD(0x0ec, storesOutput ? 0x80261041 : 0x80281342,
          @"matmul.dma_load");
    FIELD(0x0f0, storesOutput ? 0x120 : 0xce, @"matmul.dma_load_flags");
    if (!storesOutput) {
        FIELD(0x0f4, 0x600, @"matmul.left_span");
        FIELD(0x0f8, 0x180, @"matmul.right_span");
        FIELD(0x0fc, 0x135a, @"matmul.input_geometry");
        FIELD(0x100, 0x01002031, @"matmul.input_descriptor");
        FIELD(0x104, 0x803e1041, @"matmul.read_command");
        FIELD(0x108, 0x142, @"matmul.read_geometry");
        FIELD(0x10c, 0x00010140, @"matmul.read_stride");
        FIELD(0x110, 0x80, @"matmul.row_bytes");
        FIELD(0x114, 0x2000, @"matmul.left_plane_bytes");
        FIELD(0x118, 0x2000, @"matmul.right_plane_bytes");
        FIELD(0x11c, 0x2000, @"matmul.output_plane_bytes");
        FIELD(0x120, 0x80099052, @"matmul.write_command");
        FIELD(0x124, 0x140, @"matmul.write_geometry");
        FIELD(0x128, 0x8140, @"matmul.write_stride");
        FIELD(0x12c, 0x80, @"matmul.write_row_bytes");
        FIELD(0x130, 0x2000, @"matmul.write_plane_bytes");
        FIELD(0x134, 0x80049240, @"matmul.dma_inter");
        FIELD(0x138, 0x00010082, @"matmul.dma_inter_dtype");
        FIELD(0x13c, 0x00103c00, @"matmul.output_dtype");
        FIELD(0x140, 0x3000, @"matmul.sram_offset");
        FIELD(0x144, 0x21809544, @"matmul.signal");
        FIELD(0x148, 0x8000, @"matmul.signal_mask");
        FIELD(0x150, 0x23009344, @"matmul.wait");
        FIELD(0x154, 0x80, @"matmul.wait_mask");
        FIELD(0x15c, 0x22001340, @"matmul.task_begin");
        FIELD(0x160, taskStamp(task,0x21), @"matmul.task_stamp");
    } else {
        FIELD(0x0f4, 0x8140, @"matmul.read_stride");
        FIELD(0x0f8, 0x80, @"matmul.read_row_bytes");
        FIELD(0x0fc, 0x2040, @"matmul.read_plane_bytes");
        FIELD(0x100, 0x9052, @"matmul.write_command");
        FIELD(0x104, 0x14a, @"matmul.write_geometry");
        FIELD(0x108, 0x00010240, @"matmul.write_stride");
        FIELD(0x10c, 0x80049240, @"matmul.dma_inter");
        FIELD(0x110, 0x00010082, @"matmul.dma_inter_dtype");
        FIELD(0x114, 0x00103c00, @"matmul.output_dtype");
        FIELD(0x118, 0x3c00, @"matmul.output_scale");
        FIELD(0x11c, 0x80811445, @"matmul.dma_store");
        FIELD(0x120, 0x200, @"matmul.store_row_stride");
        FIELD(0x124, 0x80, @"matmul.store_rows");
        FIELD(0x128, 0x01302031, @"matmul.store_descriptor");
        FIELD(0x12c, 0x83119640, @"matmul.store_command");
        FIELD(0x130, 0x12, @"matmul.store_flags");
        FIELD(0x134, 0x00a000a0, @"matmul.store_geometry");
        FIELD(0x138, taskStamp(task,0xe0), @"matmul.store_task");
        FIELD(0x140, 0x007f0000, @"matmul.store_mask");
        FIELD(0x144, 0x20010701, @"matmul.store_surface");
        FIELD(0x148, 0x21809544, @"matmul.signal");
        FIELD(0x154, 0x23809442, @"matmul.wait");
        FIELD(0x160, 0x22001440, @"matmul.task_end");
        FIELD(0x164, 0x01000031 | (uint32_t)(task.index << 16),
              @"matmul.task_end_stamp");
    }
    return writeTaskCompletionTable(writer,base,storesOutput?0x168:0x164,
                                    task,error);
}

static BOOL writeScaleTask(H16GTDWriter *writer, NSUInteger base,
                           ANEScheduledTask *task, NSError **error) {
    uint32_t rows = (uint32_t)task.tilePlan.rows;
    FIELD(0x000, (0x22u << 16) | (uint32_t)task.index, @"scale.header");
    FIELD(0x004,1,@"scale.chain_flag");
    FIELD(0x010,0x00fff820,@"scale.dispatch_limit");
    FIELD(0x01c,3,@"scale.dependency_mode");
    FIELD(0x020,1,@"scale.source_count");
    FIELD(0x024,0x00010001,@"scale.tensor_rank");
    FIELD(0x028,rows,@"scale.rows"); FIELD(0x02c,1,@"scale.batch");
    FIELD(0x030,rows,@"scale.columns");
    FIELD(0x034,0x95458005,@"scale.compute_mode");
    FIELD(0x038,rows,@"scale.output_rows");
    FIELD(0x03c,1,@"scale.output_batch");
    FIELD(0x040,rows,@"scale.output_columns");
    FIELD(0x044,4,@"scale.head_count");
    FIELD(0x048,1,@"scale.element_bytes_log2");
    FIELD(0x04c,0x50,@"scale.vector_width");
    FIELD(0x050,6,@"scale.lane_count_code");
    FIELD(0x058,0x804c9040,@"scale.compute_command");
    FIELD(0x05c,4,@"scale.compute_argument");
    FIELD(0x060,0x120,@"scale.source_stride");
    FIELD(0x064,0x8140,@"scale.source_geometry");
    FIELD(0x068,0x80,@"scale.row_bytes");
    FIELD(0x06c,0x2000,@"scale.plane_bytes");
    FIELD(0x070,0x80049053,@"scale.dma_inter");
    FIELD(0x074,0x00010140,@"scale.dma_geometry");
    FIELD(0x078,0x10,@"scale.output_stride");
    FIELD(0x07c,0x400,@"scale.output_offset");
    FIELD(0x080,0x1140,@"scale.constant");
    FIELD(0x084,2,@"scale.mode");
    return YES;
}

static BOOL writeReduceSumTask(H16GTDWriter *writer, NSUInteger base,
                               ANEScheduledTask *task, NSError **error) {
    uint32_t rows=(uint32_t)task.tilePlan.rows;
    FIELD(0x000,(0x20u<<16)|(uint32_t)task.index,@"sum.header");
    FIELD(0x004,1,@"sum.chain_flag");
    FIELD(0x010,0x00fff820,@"sum.dispatch_limit");
    FIELD(0x01c,3,@"sum.dependency_mode");
    FIELD(0x020,1,@"sum.source_count");
    FIELD(0x024,0x00010001,@"sum.tensor_rank");
    FIELD(0x028,rows,@"sum.rows"); FIELD(0x02c,1,@"sum.batch");
    FIELD(0x030,rows,@"sum.columns");
    FIELD(0x034,0x95458005,@"sum.compute_mode");
    FIELD(0x038,rows,@"sum.output_rows");
    FIELD(0x03c,1,@"sum.output_batch");
    FIELD(0x040,rows,@"sum.output_columns");
    FIELD(0x044,4,@"sum.head_count");
    FIELD(0x048,1,@"sum.element_bytes_log2");
    FIELD(0x04c,0x50,@"sum.vector_width");
    FIELD(0x050,6,@"sum.lane_count_code");
    FIELD(0x058,0x80241041,@"sum.compute_command");
    FIELD(0x05c,0x120,@"sum.compute_argument");
    FIELD(0x060,0x80,@"sum.source_stride");
    FIELD(0x064,0x2050,@"sum.source_geometry");
    FIELD(0x068,0x80049053,@"sum.dma_inter");
    FIELD(0x06c,0x00010240,@"sum.dma_geometry");
    FIELD(0x070,0x10,@"sum.output_stride");
    FIELD(0x074,0x400,@"sum.output_bytes");
    FIELD(0x078,0x1140,@"sum.output_offset");
    FIELD(0x07c,0x1000,@"sum.sram_offset");
    return YES;
}

static BOOL writeReduceTask(H16GTDWriter *writer, NSUInteger base,
                            ANEScheduledTask *task, BOOL reciprocal,
                            NSError **error) {
    uint32_t rows = (uint32_t)task.tilePlan.rows;
    FIELD(0x000, (0x24u - (reciprocal ? 2u : 0u)) << 16 |
                 (uint32_t)task.index, @"reduce.header");
    FIELD(0x010, 0x00fff820, @"reduce.dispatch_limit");
    FIELD(0x01c, 3, @"reduce.dependency_mode");
    FIELD(0x020, 1, @"reduce.source_count");
    FIELD(0x024, 0x00010001, @"reduce.tensor_rank");
    FIELD(0x028, rows, @"reduce.rows");
    FIELD(0x02c, 4, @"reduce.head_count");
    FIELD(0x030, rows, @"reduce.columns");
    FIELD(0x034, 0x9d418005, @"reduce.compute_mode");
    FIELD(0x038, rows, @"reduce.output_rows");
    FIELD(0x03c, 4, @"reduce.output_heads");
    FIELD(0x040, rows, @"reduce.output_columns");
    FIELD(0x044, 4, @"reduce.axis");
    FIELD(0x048, 0x30, @"reduce.vector_width");
    FIELD(0x04c, 0x25, @"reduce.operation");
    FIELD(0x050, reciprocal ? 1 : 0x10, @"reduce.lane_count");
    FIELD(0x058, reciprocal ? 0x818e1041 : 0x80e71042,
          @"reduce.compute_command");
    FIELD(0x05c, 0x00400100, @"reduce.compute_geometry");
    FIELD(0x060, reciprocal ? 0x00010240 : 0x8140, @"reduce.source_stride");
    FIELD(0x064, reciprocal ? 0x10 : 0x80, @"reduce.source_rows");
    FIELD(0x068, reciprocal ? 0x400 : 0x2000, @"reduce.source_bytes");
    FIELD(0x06c, reciprocal ? 0x80 : 0x00010140, @"reduce.output_rows");
    FIELD(0x070, reciprocal ? 0x2050 : 0x10, @"reduce.output_geometry");
    FIELD(0x074, reciprocal ? 0x9054 : 0x400, @"reduce.output_command");
    FIELD(0x078, reciprocal ? 0x80 : 0x9054, @"reduce.output_stride");
    FIELD(0x07c, reciprocal ? 0x2050 : 0x80, @"reduce.output_bytes");
    FIELD(0x080, reciprocal ? 0x1140 : 0x2050, @"reduce.output_offset");
    FIELD(0x084, reciprocal ? 0x00080004 : 0x80011140,
          @"reduce.finalize");
    if (!reciprocal) {
        FIELD(0x088, 0x000c0000, @"reduce.identity_kind");
        FIELD(0x08c, 0x3fb8a000, @"reduce.identity_value");
    }
    return YES;
}

static BOOL writeCenterExpTask(H16GTDWriter *writer, NSUInteger base,
                               ANEScheduledTask *task, NSError **error) {
    uint32_t rows = (uint32_t)task.tilePlan.rows;
    FIELD(0x000, (0x3bu << 16) | (uint32_t)task.index, @"lut.header");
    FIELD(0x010, 0x00fff820, @"lut.dispatch_limit");
    FIELD(0x01c, 3, @"lut.dependency_mode");
    FIELD(0x020, 0x00050001, @"lut.control");
    FIELD(0x024, 0xffc01540, @"lut.control_mask");
    FIELD(0x028, 0x00060240, @"lut.task_control");
    for (NSUInteger offset=0x02c;offset<=0x04c;offset+=4)
        FIELD(offset,0x00050020,@"lut.chain_slot");
    FIELD(0x050, 0x00031551, @"lut.barrier");
    for (NSUInteger offset=0x054;offset<=0x06c;offset+=4)
        FIELD(offset,0x00050020,@"lut.chain_slot");
    FIELD(0x070, 0x9584, @"lut.table_selector");
    FIELD(0x074, 0x00050021, @"lut.chain_end");
    FIELD(0x078, 0x80, @"lut.table_bytes");
    FIELD(0x07c, 0x00010001, @"lut.tensor_rank");
    FIELD(0x080, rows, @"lut.rows");
    FIELD(0x084, 4, @"lut.head_count");
    FIELD(0x088, rows, @"lut.columns");
    FIELD(0x08c, 0x93618005, @"lut.compute_mode");
    FIELD(0x090, rows, @"lut.output_rows");
    FIELD(0x094, 4, @"lut.output_heads");
    FIELD(0x098, rows, @"lut.output_columns");
    FIELD(0x09c, 0x00014000, @"lut.table_mode");
    FIELD(0x0a0, 4, @"lut.rank");
    FIELD(0x0a4, 0x00222200, @"lut.axis_strides");
    FIELD(0x0ac, 0x00100000, @"lut.surface_limit");
    FIELD(0x0b0, 0x800c1041, @"lut.read_command");
    FIELD(0x0b4, 0x108, @"lut.read_geometry");
    FIELD(0x0b8, 0x80, @"lut.read_row_bytes");
    FIELD(0x0bc, 0x2050, @"lut.read_stride");
    FIELD(0x0c0, 0x80031052, @"lut.write_command");
    FIELD(0x0c4, 0x140, @"lut.write_geometry");
    FIELD(0x0c8, 0x80, @"lut.write_row_bytes");
    FIELD(0x0cc, 0x2050, @"lut.write_stride");
    FIELD(0x0d0, 0x80049240, @"lut.dma_inter");
    FIELD(0x0d4, 0x00010082, @"lut.dma_inter_dtype");
    FIELD(0x0d8, 0x00123c0c, @"lut.output_dtype");
    FIELD(0x0dc, 0x3c00, @"lut.output_scale");
    FIELD(0x0e0, 0x24009586, @"lut.kernel_relocation");
    return YES;
}

static BOOL writeNormalizeTransposeTask(H16GTDWriter *writer, NSUInteger base,
                                        ANEScheduledTask *task,
                                        NSError **error) {
    uint32_t rows=(uint32_t)task.tilePlan.rows;
    FIELD(0x000,(0x22u<<16)|(uint32_t)task.index,@"normalize.header");
    FIELD(0x010,0x20,@"normalize.dependency_mask");
    FIELD(0x01c,3,@"normalize.dependency_mode");
    FIELD(0x020,0x00050001,@"normalize.control");
    FIELD(0x024,0x00010001,@"normalize.tensor_rank");
    FIELD(0x028,rows,@"normalize.rows");
    FIELD(0x02c,4,@"normalize.head_count");
    FIELD(0x030,rows,@"normalize.columns");
    FIELD(0x034,0x93618005,@"normalize.compute_mode");
    FIELD(0x038,rows,@"normalize.output_rows");
    FIELD(0x03c,4,@"normalize.output_heads");
    FIELD(0x040,rows,@"normalize.output_columns");
    FIELD(0x044,0x00044000,@"normalize.permutation");
    FIELD(0x048,4,@"normalize.rank");
    FIELD(0x04c,0x30222200,@"normalize.axis_strides");
    FIELD(0x050,2,@"normalize.element_bytes");
    FIELD(0x054,0x00100000,@"normalize.surface_limit");
    FIELD(0x058,0x800c1041,@"normalize.read_command");
    FIELD(0x05c,0x108,@"normalize.read_geometry");
    FIELD(0x060,0x80,@"normalize.read_row_bytes");
    FIELD(0x064,0x2050,@"normalize.read_stride");
    FIELD(0x068,0x00019052,@"normalize.write_command");
    FIELD(0x06c,0x140,@"normalize.write_geometry");
    FIELD(0x070,0x8140,@"normalize.write_stride");
    FIELD(0x074,0x80,@"normalize.write_row_bytes");
    FIELD(0x078,0x2040,@"normalize.write_plane_bytes");
    FIELD(0x07c,0x80021241,@"normalize.dma_inter");
    FIELD(0x080,0x00103c0c,@"normalize.output_dtype");
    FIELD(0x084,0x3c00,@"normalize.output_scale");
    return YES;
}

#undef FIELD

@implementation H16GMixedTaskEncoding
- (instancetype)initWithTDProgram:(H16GEncodedTDProgram *)tdProgram
                   constantRegion:(NSData *)constantRegion
                scratchByteLength:(NSUInteger)scratchByteLength {
    self = [super init];
    if (self) {
        _tdProgram = tdProgram;
        _constantRegion = [constantRegion copy];
        _scratchByteLength = scratchByteLength;
    }
    return self;
}
@end

@implementation H16GMixedTaskEncoder
+ (H16GMixedTaskEncoding *)encodeGraph:(ANEOperationGraph *)graph
                               scheduled:(ANEScheduledGraph *)scheduled
                                   error:(NSError **)error {
    if (![H16GDecodedFormValidator validateMixedGraph:graph
        scheduled:scheduled error:error]) return nil;

    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:0x948];
    NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
    NSUInteger base=0;
    for(ANEScheduledTask *task in scheduled.tasks) {
        H16GDecodedTaskForm form=decodedForm(graph,task);
        BOOL encoded=NO;
        switch(form) {
            case H16GDecodedTaskFormSplit:
                encoded=writeSplitTask(writer,base,task,graph,error); break;
            case H16GDecodedTaskFormTransposePair:
                encoded=writeTransposeTask(writer,base,task,error); break;
            case H16GDecodedTaskFormMatmul:
                encoded=writeMatmulTask(writer,base,task,error); break;
            case H16GDecodedTaskFormScale:
                encoded=writeScaleTask(writer,base,task,error); break;
            case H16GDecodedTaskFormReduceMax:
                encoded=writeReduceTask(writer,base,task,NO,error); break;
            case H16GDecodedTaskFormCenterExp:
                encoded=writeCenterExpTask(writer,base,task,error);
                if(encoded) [relocations addObject:@(base+0xe4)];
                break;
            case H16GDecodedTaskFormReduceSum:
                encoded=writeReduceSumTask(writer,base,task,error); break;
            case H16GDecodedTaskFormReciprocal:
                encoded=writeReduceTask(writer,base,task,YES,error); break;
            case H16GDecodedTaskFormNormalizeTranspose:
                encoded=writeNormalizeTransposeTask(writer,base,task,error);
                break;
            case H16GDecodedTaskFormUnknown: break;
        }
        if(!encoded) {
            if(error && !*error) *error=[NSError
                errorWithDomain:H16GTDWriterErrorDomain code:6
                userInfo:@{NSLocalizedDescriptionKey:
                    @"scheduled task has no decoded H16G field encoder"}];
            return nil;
        }
        base+=encodedSize(form,task);
    }
    if(base!=writer.data.length) {
        if(error) *error=[NSError errorWithDomain:H16GTDWriterErrorDomain code:7
            userInfo:@{NSLocalizedDescriptionKey:
                @"encoded H16G task sizes do not fill the declared TD stream"}];
        return nil;
    }
    H16GEncodedTDProgram *program = [[H16GEncodedTDProgram alloc]
        initWithData:writer.data kernelRelocationOffsets:relocations
        programRecordCount:0x75 programFormatCode:1];
    NSData *constants = [H16GLUTEncoder
        constantRegionForOperationName:@"exp" error:error];
    if (!constants) return nil;
    return [[H16GMixedTaskEncoding alloc] initWithTDProgram:program
        constantRegion:constants scratchByteLength:0x10000];
}
@end
