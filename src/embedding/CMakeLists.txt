target_sources(
  sirius_objects
  PRIVATE control.cpp
          execution_stats.cpp
          input.cpp
          tae_demand.cpp
          result.cpp
          result_codec.cpp
          result_gpu.cpp
          native_gpu.cpp
          plan_bindings.cpp)
