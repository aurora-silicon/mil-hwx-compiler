UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
CXX := clang++
CXXFLAGS := -O2 -std=c++17 -fobjc-arc -Wall -Wextra -Werror -Iinclude
FRAMEWORKS := -framework Foundation
else
GNUSTEP_PREFIX ?= $(HOME)/.local/mil-hwx-gnustep
GNUSTEP_CONFIG ?= $(GNUSTEP_PREFIX)/bin/gnustep-config
ifeq ($(origin CXX),default)
CXX := $(or $(firstword $(wildcard /usr/bin/clang++-18 /usr/bin/clang++-17 /usr/bin/clang++-16 /usr/bin/clang++-15 /usr/bin/clang++-14)),$(shell command -v clang++ 2>/dev/null))
endif
GNUSTEP_FLAGS := $(shell GNUSTEP_CONFIG_FILE=$(GNUSTEP_PREFIX)/etc/GNUstep/GNUstep.conf $(GNUSTEP_CONFIG) --objc-flags 2>/dev/null)
CXXFLAGS := -std=c++17 -fobjc-arc $(GNUSTEP_FLAGS) -Wall -Wextra -Werror -Iinclude
FRAMEWORKS := $(shell GNUSTEP_CONFIG_FILE=$(GNUSTEP_PREFIX)/etc/GNUstep/GNUstep.conf $(GNUSTEP_CONFIG) --base-libs 2>/dev/null)
endif
BUILD := build

SUPPORT_SOURCES := lib/Support/ANEDiagnostic.mm
MIL_LEXER_SOURCES := lib/MIL/MILLexer.mm
MIL_PARSER_SOURCES := lib/MIL/MILSyntax.mm lib/MIL/MILParser.mm lib/MIL/MILPrinter.mm
GRAPH_SOURCES := lib/IR/ANEGraphIR.mm lib/IR/ANEGraphVerifier.mm lib/MIL/MILGraphImporter.mm
OP_GRAPH_SOURCES := lib/IR/ANEOperationGraph.mm
TRANSFORM_SOURCES := lib/Transform/ANENormalizePass.mm lib/Transform/ANEDecomposePass.mm
FUSION_SOURCES := lib/Transform/ANEFusionPass.mm plugins/H16G/H16GTarget.mm
LEGALIZE_SOURCES := lib/Transform/ANEH16GLegalizePass.mm
SCHEDULED_SOURCES := lib/IR/ANEScheduledGraph.mm lib/Planning/ANETilePlanner.mm lib/Planning/ANEMemoryPlanner.mm lib/Planning/ANEComposedRegionPlanner.mm lib/Planning/ANEProgramPartition.mm lib/Planning/ANETaskScheduler.mm
STRUCTURED_TD_SOURCES := plugins/H16G/Encoding/H16GTDWriter.mm plugins/H16G/Encoding/H16GConvEncoder.mm plugins/H16G/Encoding/H16GConvChainEncoder.mm plugins/H16G/Encoding/H16GDepthwiseEncoder.mm plugins/H16G/Encoding/H16GRegularConvEncoder.mm plugins/H16G/Encoding/H16GMatmulEncoder.mm plugins/H16G/Encoding/H16GALUEncoder.mm plugins/H16G/Encoding/H16GBroadcastALUEncoder.mm plugins/H16G/Encoding/H16GMatrixRowDivisionEncoder.mm plugins/H16G/Encoding/H16GLUTEncoder.mm plugins/H16G/Encoding/H16GReduceEncoder.mm plugins/H16G/Encoding/H16GSRAMChainEncoder.mm plugins/H16G/Encoding/H16GLayoutEncoder.mm plugins/H16G/Encoding/H16GLayoutConvChainEncoder.mm plugins/H16G/Encoding/H16GDecodedFormValidator.mm plugins/H16G/Encoding/H16GMixedTaskEncoder.mm
TASK_ENCODING_SOURCES := plugins/H16G/Encoding/H16GEncodedTask.mm plugins/H16G/Encoding/H16GTaskEncoder.mm plugins/H16G/Encoding/H16GTaskComposer.mm
CONSTANT_PACKER_SOURCES := plugins/H16G/Encoding/H16GConstantPacker.mm
OBJECT_WRITER_SOURCES := lib/HWX/HWXObjectWriter.mm
STAGED_DRIVER_SOURCES := lib/Driver/ANEStagedCompiler.mm plugins/H16G/Encoding/H16GProgramEncoder.mm plugins/H16G/Encoding/H16GProgramAssembler.mm $(TASK_ENCODING_SOURCES)
PASS_SOURCES := lib/Pass/ANEPassManager.mm
PLUGIN_SOURCES := lib/Plugin/ANEPluginRegistry.mm
DRIVER_SOURCES := lib/Model/ANEBlobResolver.mm lib/Runtime/ANEExecutableBundle.mm lib/Driver/ANECompiler.mm
PRODUCTION_COMPILER_SOURCES := $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) $(SCHEDULED_SOURCES) $(STRUCTURED_TD_SOURCES) $(OBJECT_WRITER_SOURCES) $(CONSTANT_PACKER_SOURCES) $(STAGED_DRIVER_SOURCES) lib/Driver/ANECompiler.mm lib/Model/ANEBlobResolver.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/Runtime/ANEExecutableBundle.mm
RUNTIME_SOURCES := lib/Runtime/ANEProvisionedRuntime.mm
RUNTIME_BUNDLE_SOURCES := lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/IR/ANEGraphIR.mm lib/Runtime/ANEExecutableBundle.mm $(RUNTIME_SOURCES)
RUNTIME_INCLUDES := -Ilib/IR -Ilib/HWX -Ilib/Runtime
BENCHMARK_STATS_SOURCES := lib/Runtime/ANEBenchmarkStats.cpp
H13_SOURCES := plugins/H14LayerLink.cpp plugins/LayerOps.cpp plugins/H13/ANEH13Compiler.mm plugins/H13/H13Program.cpp plugins/H13/H13ANEC.cpp
H14_SOURCES := plugins/H14/ANEH14Compiler.mm plugins/H14/H14Program.cpp

.PHONY: all test clean test-cli test-no-pattern-shortcuts

all: test

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_diagnostics: tests/test_diagnostics.mm $(SUPPORT_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_benchmark_stats: tests/test_benchmark_stats.cpp lib/Runtime/ANEBenchmarkStats.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/test_mil_lexer: tests/test_mil_lexer.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_mil_parser: tests/test_mil_parser.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_graph_import: tests/test_graph_import.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_operation_graph: tests/test_operation_graph.mm $(SUPPORT_SOURCES) lib/IR/ANEGraphIR.mm $(OP_GRAPH_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/IR $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_graph_transforms: tests/test_graph_transforms.mm $(SUPPORT_SOURCES) lib/IR/ANEGraphIR.mm $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/IR -Ilib/Transform $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_structural_fusion: tests/test_structural_fusion.mm $(SUPPORT_SOURCES) lib/IR/ANEGraphIR.mm $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/IR -Ilib/Transform -Iplugins/H16G $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_h16g_legalization: tests/test_h16g_legalization.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Iplugins/H16G $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_hwx_planning: tests/test_hwx_planning.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) $(SCHEDULED_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Iplugins/H16G $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_structured_td_encoding: tests/test_structured_td_encoding.mm $(SUPPORT_SOURCES) lib/HWX/HWXImage.mm plugins/H16G/H16GTarget.mm $(STRUCTURED_TD_SOURCES) $(TASK_ENCODING_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/HWX -Ilib/IR -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_program_partition: tests/test_program_partition.mm lib/Planning/ANEProgramPartition.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/Planning $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_program_composition: tests/test_program_composition.mm plugins/H16G/H16GTarget.mm $(STRUCTURED_TD_SOURCES) $(TASK_ENCODING_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/IR -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_hwx_object_writer: tests/test_hwx_object_writer.mm lib/HWX/HWXImage.mm $(OBJECT_WRITER_SOURCES) $(STRUCTURED_TD_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/HWX -Ilib/IR -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_h16g_constant_packing: tests/test_h16g_constant_packing.mm $(CONSTANT_PACKER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_staged_conv_compiler: tests/test_staged_conv_compiler.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) $(SCHEDULED_SOURCES) $(STRUCTURED_TD_SOURCES) $(OBJECT_WRITER_SOURCES) $(CONSTANT_PACKER_SOURCES) $(STAGED_DRIVER_SOURCES) lib/Model/ANEBlobResolver.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/Runtime/ANEExecutableBundle.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/Model -Ilib/HWX -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_pass_manager: tests/test_pass_manager.mm $(SUPPORT_SOURCES) $(PASS_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/Pass $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_plugin_registry: tests/test_plugin_registry.mm $(SUPPORT_SOURCES) $(PLUGIN_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/Plugin $^ $(FRAMEWORKS) -o $@

$(BUILD)/h13_exec: tests/hardware/run_h13.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/conv_exec: tests/hardware/run_conv_relu.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/layout_exec: tests/hardware/run_layout.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/layout_conv_exec: tests/hardware/run_layout_conv.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/depthwise_exec: tests/hardware/run_depthwise.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/regular_conv_exec: tests/hardware/run_regular_conv.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/matmul_exec: tests/hardware/run_matmul.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/alu_exec: tests/hardware/run_alu.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/unary_exec: tests/hardware/run_unary.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/reduce_exec: tests/hardware/run_reduce.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/broadcast_alu_exec: tests/hardware/run_broadcast_alu.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/online_reduction_exec: tests/hardware/run_online_reduction.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/affine_scan_exec: tests/hardware/run_affine_scan.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/chunked_deltanet_exec: tests/hardware/run_chunked_deltanet.mm tests/hardware/ANEAppleBaselineRuntime.mm $(BENCHMARK_STATS_SOURCES) $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) -Itests/hardware $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/matmul_gelu_exec: tests/hardware/run_matmul_gelu.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/compiler_ab_benchmark: tests/hardware/benchmark_compiler_ab.mm tests/hardware/ANEAppleBaselineRuntime.mm $(BENCHMARK_STATS_SOURCES) $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) -Itests/hardware $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/profile_program_chain: tests/hardware/profile_program_chain.mm $(BENCHMARK_STATS_SOURCES) $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/scalar_fold_exec: tests/hardware/run_scalar_fold.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/prepare_broadcast_alu: tests/hardware/prepare_broadcast_alu.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/HWX/HWXObjectWriter.mm lib/IR/ANEGraphIR.mm lib/Runtime/ANEExecutableBundle.mm plugins/H16G/Encoding/H16GTDWriter.mm plugins/H16G/Encoding/H16GConvChainEncoder.mm plugins/H16G/Encoding/H16GBroadcastALUEncoder.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/HWX -Ilib/IR -Ilib/Runtime -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_layout: tests/hardware/prepare_layout.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_layout_conv: tests/hardware/prepare_layout_conv.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_depthwise: tests/hardware/prepare_depthwise.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_regular_conv: tests/hardware/prepare_regular_conv.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_matmul: tests/hardware/prepare_matmul.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_alu: tests/hardware/prepare_alu.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_unary: tests/hardware/prepare_unary.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_reduce: tests/hardware/prepare_reduce.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/attention_exec: tests/hardware/run_attention.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/w8a8_exec: tests/hardware/run_w8a8.mm $(RUNTIME_BUNDLE_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(RUNTIME_INCLUDES) $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

$(BUILD)/prepare_w8a8_model: tests/hardware/prepare_w8a8_model.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_clean_conv: tests/hardware/prepare_clean_conv.mm lib/HWX/ANEHWXArtifact.mm lib/Runtime/ANEExecutableBundle.mm lib/HWX/HWXImage.mm $(OBJECT_WRITER_SOURCES) $(STRUCTURED_TD_SOURCES) $(CONSTANT_PACKER_SOURCES) lib/IR/ANEGraphIR.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/HWX -Ilib/IR -Ilib/Runtime -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_staged_conv: tests/hardware/prepare_staged_conv.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) $(SCHEDULED_SOURCES) $(STRUCTURED_TD_SOURCES) $(OBJECT_WRITER_SOURCES) $(CONSTANT_PACKER_SOURCES) $(STAGED_DRIVER_SOURCES) lib/Model/ANEBlobResolver.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/Runtime/ANEExecutableBundle.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/Model -Ilib/HWX -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_staged_w8a8: tests/hardware/prepare_staged_w8a8.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) $(SCHEDULED_SOURCES) $(STRUCTURED_TD_SOURCES) $(OBJECT_WRITER_SOURCES) $(CONSTANT_PACKER_SOURCES) $(STAGED_DRIVER_SOURCES) lib/Model/ANEBlobResolver.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/Runtime/ANEExecutableBundle.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/Model -Ilib/HWX -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/prepare_staged_attention: tests/hardware/prepare_staged_attention.mm $(SUPPORT_SOURCES) $(MIL_LEXER_SOURCES) $(MIL_PARSER_SOURCES) $(GRAPH_SOURCES) $(OP_GRAPH_SOURCES) $(TRANSFORM_SOURCES) $(FUSION_SOURCES) $(LEGALIZE_SOURCES) $(SCHEDULED_SOURCES) $(STRUCTURED_TD_SOURCES) $(OBJECT_WRITER_SOURCES) $(CONSTANT_PACKER_SOURCES) $(STAGED_DRIVER_SOURCES) lib/Model/ANEBlobResolver.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/Runtime/ANEExecutableBundle.mm | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/Model -Ilib/HWX -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/test_compiler_e2e: tests/test_compiler_e2e.mm $(PRODUCTION_COMPILER_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding $^ $(FRAMEWORKS) -o $@

$(BUILD)/mil-hwxc: tools/mil-hwxc.mm $(PRODUCTION_COMPILER_SOURCES) $(H13_SOURCES) $(H14_SOURCES) plugins/LayerOps.h plugins/LayerGraph.h plugins/NativeLayerPlan.h plugins/NativeLayerTemplates.inc plugins/H13/ANEH13Compiler.h plugins/H13/H13Program.h $(wildcard plugins/H13/*.inc) plugins/H14/ANEH14Compiler.h plugins/H14/H14Program.h $(wildcard plugins/H14/*.inc) $(wildcard plugins/H17/*.inc) $(wildcard plugins/H18/*.inc) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/MIL -Ilib/IR -Ilib/Transform -Ilib/Planning -Ilib/Driver -Ilib/HWX -Ilib/Model -Ilib/Runtime -Iplugins/H16G -Iplugins/H16G/Encoding -Iplugins/H13 -Iplugins/H14 -Iplugins/H17 -Iplugins/H18 $(filter %.mm %.cpp,$^) $(FRAMEWORKS) -o $@

test-cli: $(BUILD)/mil-hwxc
	bash tests/test_cli.sh

test-no-pattern-shortcuts: $(BUILD)/mil-hwxc
	bash tests/test_no_pattern_shortcuts.sh

$(BUILD)/test_runtime_contract: tests/test_runtime_contract.mm lib/HWX/ANEHWXArtifact.mm lib/HWX/HWXImage.mm lib/IR/ANEGraphIR.mm lib/Runtime/ANEExecutableBundle.mm $(RUNTIME_SOURCES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ilib/IR -Ilib/HWX -Ilib/Runtime $^ $(FRAMEWORKS) -framework IOSurface -ldl -o $@

test: $(BUILD)/test_diagnostics $(BUILD)/test_benchmark_stats $(BUILD)/test_mil_lexer $(BUILD)/test_mil_parser $(BUILD)/test_graph_import $(BUILD)/test_operation_graph $(BUILD)/test_graph_transforms $(BUILD)/test_structural_fusion $(BUILD)/test_h16g_legalization $(BUILD)/test_hwx_planning $(BUILD)/test_structured_td_encoding $(BUILD)/test_program_partition $(BUILD)/test_program_composition $(BUILD)/test_hwx_object_writer $(BUILD)/test_h16g_constant_packing $(BUILD)/test_staged_conv_compiler $(BUILD)/test_pass_manager $(BUILD)/test_plugin_registry $(BUILD)/test_compiler_e2e $(BUILD)/test_runtime_contract test-cli test-no-pattern-shortcuts
	$(BUILD)/test_diagnostics
	$(BUILD)/test_benchmark_stats
	$(BUILD)/test_mil_lexer
	$(BUILD)/test_mil_parser
	$(BUILD)/test_graph_import
	$(BUILD)/test_operation_graph
	$(BUILD)/test_graph_transforms
	$(BUILD)/test_structural_fusion
	$(BUILD)/test_h16g_legalization
	$(BUILD)/test_hwx_planning
	$(BUILD)/test_structured_td_encoding
	$(BUILD)/test_program_partition
	$(BUILD)/test_program_composition
	$(BUILD)/test_hwx_object_writer
	$(BUILD)/test_h16g_constant_packing
	$(BUILD)/test_staged_conv_compiler
	$(BUILD)/test_pass_manager
	$(BUILD)/test_plugin_registry
	$(BUILD)/test_compiler_e2e
	$(BUILD)/test_runtime_contract
	bash tests/test_release_hygiene.sh

.PHONY: test-h13
test-h13: $(BUILD)/test_h13_encoding $(BUILD)/test_h13_anec $(BUILD)/mil-hwxc
	$(BUILD)/test_h13_encoding
	$(BUILD)/test_h13_anec
	python3 tests/test_h13_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_split_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_layout_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_composite_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_batched_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_registry_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_boolean_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_tile_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_linear_tiling_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_matmul_envelope_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_silu_envelope_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_norm_envelope_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_conv_envelope_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_generic_conv_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_camera_conv_cli.py $(BUILD)/mil-hwxc
	python3 research/generate_layer_templates.py --check
	python3 tests/test_layer_oracle_policy.py --target h13
	python3 tests/test_native_layer_cli.py $(BUILD)/mil-hwxc --target h13
	python3 tests/test_native_linked_cli.py $(BUILD)/mil-hwxc --target h13
	python3 tests/test_native_graph_cli.py $(BUILD)/mil-hwxc --target H13
	python3 tests/test_h13_linked_cli.py $(BUILD)/mil-hwxc
	python3 research/h13_l2_edge_parity.py $(BUILD)/mil-hwxc research/oracles/h13-l2-edges/*.json
	python3 tests/test_h13_select_envelope_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_chain_envelope_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_transpose_cli.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_slice_cli.py $(BUILD)/mil-hwxc
	python3 research/mint_conv_probes.py --check
.PHONY: test-layer-reference
test-layer-reference:
	python3 tests/test_layer_reference.py

.PHONY: test-h13-reference
test-h13-reference: $(BUILD)/mil-hwxc
	python3 tests/test_h13_reference.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_deadline.py
.PHONY: test-h13-simulation
test-h13-simulation: $(BUILD)/mil-hwxc
	python3 tests/test_h13_package_simulation.py
.PHONY: test-h13-parity
test-h13-parity: $(BUILD)/mil-hwxc
	python3 tests/test_h13_parity.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_fixture_programs.py $(BUILD)/mil-hwxc
	python3 tests/test_h13_matvec_programs.py $(BUILD)/mil-hwxc
	python3 tests/test_libane_overrun.py $(BUILD)/mil-hwxc
.PHONY: test-h14-parity
test-h14-parity: $(BUILD)/mil-hwxc
	python3 tests/test_layer_oracle_policy.py --target h14
	python3 tests/test_h13_camera_conv_cli.py $(BUILD)/mil-hwxc --target H14
	python3 research/generate_layer_templates.py --check
	python3 tests/test_native_layer_cli.py $(BUILD)/mil-hwxc --target h14
	python3 tests/test_native_linked_cli.py $(BUILD)/mil-hwxc --target h14
	python3 tests/test_native_graph_cli.py $(BUILD)/mil-hwxc --target H14
	python3 research/generate_h14_templates.py --check
	python3 tests/test_h14_parity.py $(BUILD)/mil-hwxc
	python3 tests/test_h14_fixture_programs.py $(BUILD)/mil-hwxc
	python3 tests/test_h14_matvec_programs.py $(BUILD)/mil-hwxc
	python3 tests/test_h14_matvec_row_programs.py $(BUILD)/mil-hwxc
	python3 tests/test_h14_unary_programs.py $(BUILD)/mil-hwxc
	python3 tests/test_exp_scaled_chain_case.py
	python3 tests/test_h14_exp_chain_cli.py $(BUILD)/mil-hwxc

.PHONY: test-h14-exp-model
test-h14-exp-model:
	python3 tests/test_h14_exp_arithmetic.py

.PHONY: test-h17-parity test-h18-parity
test-h17-parity: $(BUILD)/mil-hwxc
	python3 research/generate_h14_templates.py --check
	python3 tests/test_h17_parity.py $(BUILD)/mil-hwxc H17

test-h18-parity: $(BUILD)/mil-hwxc
	python3 research/generate_h14_templates.py --check
	python3 tests/test_h17_parity.py $(BUILD)/mil-hwxc H18

$(BUILD)/test_h13_encoding: plugins/H13/H13Program.cpp tests/test_h13_encoding.cpp plugins/H13/H13Program.h plugins/LayerGraph.h | $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror $(filter %.cpp,$^) -o $@

$(BUILD)/test_h13_anec: plugins/H13/H13ANEC.cpp tests/test_h13_anec.cpp plugins/H13/H13Program.h | $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror $(filter %.cpp,$^) -o $@

.PHONY: test-hwx-inspection
test-hwx-inspection:
	python3 tests/test_hwx_inspection.py

test: test-h13 test-h13-parity test-h13-reference test-h13-simulation test-h14-parity test-h17-parity test-h18-parity test-hwx-inspection

.PHONY: test-h13-hardware
test-h13-hardware: $(BUILD)/mil-hwxc $(BUILD)/h13_exec
	@test -n "$(H13_MIL)" -a -n "$(H13_MODEL_ROOT)" -a -n "$(H13_INPUTS)" || \
		(echo 'usage: make test-h13-hardware H13_MIL=model.mil H13_MODEL_ROOT=models H13_INPUTS="x=input.fp16"' >&2; exit 64)
	tests/run_h13_hardware.sh "$(H13_MIL)" "$(H13_MODEL_ROOT)" $(H13_INPUTS)

clean:
	rm -rf $(BUILD)
