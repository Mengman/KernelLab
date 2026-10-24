add_executable(runtime_tests unit/runtime_tests.cpp)
target_link_libraries(runtime_tests PRIVATE gpulab::runtime)
kernellab_target_options(runtime_tests)
add_test(NAME runtime.unit COMMAND runtime_tests)
set_tests_properties(runtime.unit PROPERTIES LABELS cpu)

add_executable(random_tests unit/random_tests.cpp)
target_link_libraries(random_tests PRIVATE gpulab::test_support)
kernellab_target_options(random_tests)
add_test(NAME testing.random COMMAND random_tests)
set_tests_properties(testing.random PROPERTIES LABELS cpu TIMEOUT 30)

add_executable(bench_stats_tests unit/bench_stats_tests.cpp)
target_link_libraries(bench_stats_tests PRIVATE gpulab::bench_stats)
kernellab_target_options(bench_stats_tests)
add_test(NAME bench.stats COMMAND bench_stats_tests)
set_tests_properties(bench.stats PROPERTIES LABELS cpu)

if(TARGET tiny_transformer_model)
  add_executable(transformer_config_tests unit/transformer_config_tests.cpp)
  target_link_libraries(transformer_config_tests PRIVATE gpulab::transformer)
  kernellab_target_options(transformer_config_tests)
  add_test(NAME transformer.config COMMAND transformer_config_tests)
  set_tests_properties(transformer.config PROPERTIES LABELS cpu)
endif()

foreach(tool IN ITEMS device_inspect gpulab_bench tiny_transformer)
  if(TARGET ${tool})
    add_test(NAME ${tool}.help COMMAND ${tool} --help)
    add_test(NAME ${tool}.reject_unknown COMMAND ${tool} --invalid-argument)
    set_tests_properties(${tool}.help ${tool}.reject_unknown PROPERTIES LABELS cpu)
    set_tests_properties(${tool}.reject_unknown PROPERTIES WILL_FAIL TRUE)
  endif()
endforeach()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  foreach(pair IN ITEMS "device;device_inspect" "model;tiny_transformer" "bench;gpulab_bench")
    list(GET pair 0 mode)
    list(GET pair 1 tool)
    if(TARGET ${tool})
      add_test(NAME ${tool}.contract
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cli_contract.py"
          ${mode} "$<TARGET_FILE:${tool}>")
      set_tests_properties(${tool}.contract PROPERTIES LABELS integration TIMEOUT 30)
    endif()
  endforeach()
else()
  message(STATUS "Python3 not found: C++ tests remain enabled; JSON CLI contract tests omitted.")
endif()

if(KERNELLAB_ENABLE_GPU_TESTS)
  add_executable(cuda_smoke_tests gpu/cuda_smoke_tests.cpp)
  target_link_libraries(cuda_smoke_tests PRIVATE gpulab::runtime)
  kernellab_target_options(cuda_smoke_tests)
  add_test(NAME cuda.smoke COMMAND cuda_smoke_tests)
  set_tests_properties(cuda.smoke PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 30)
endif()

# Assignment tests: the sources and targets live in benchmarks/pXX/, while the
# CTest registration stays here so the device-free suite runs on every host and
# the device suite stays selectable with -L gpu. Test names follow the
# pXX.cpu / pXX.gpu pattern. Keep the TARGET guard: a CPU-only configure must
# not register targets it did not build, otherwise ctest runs stale commands.
if(TARGET p02_test_axpby)
  add_test(NAME p02.cpu COMMAND p02_test_axpby)
  set_tests_properties(p02.cpu PROPERTIES LABELS cpu TIMEOUT 30)
endif()

if(KERNELLAB_ENABLE_GPU_TESTS AND TARGET p01_test_fma)
  add_test(NAME p01.gpu COMMAND p01_test_fma)
  set_tests_properties(p01.gpu PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()


if(KERNELLAB_ENABLE_GPU_TESTS AND TARGET p02_test_axpby_gpu)
  add_test(NAME p02.gpu COMMAND p02_test_axpby_gpu)
  set_tests_properties(p02.gpu PROPERTIES LABELS gpu SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
