function(kernellab_target_options target)
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>"
  )
endfunction()
