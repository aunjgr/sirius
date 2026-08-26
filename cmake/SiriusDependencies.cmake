# Resolve the TAE scanner in both supported layouts: Sirius as an extension
# inside mo-sirius-sidecar, or a standalone recursive Sirius checkout.
set(SIRIUS_TAE_SCANNER_DIR "${CMAKE_CURRENT_LIST_DIR}/../../tae-scanner")
if(NOT EXISTS "${SIRIUS_TAE_SCANNER_DIR}/CMakeLists.txt")
  set(SIRIUS_TAE_SCANNER_DIR "${CMAKE_CURRENT_LIST_DIR}/../tae-scanner")
endif()
if(NOT EXISTS "${SIRIUS_TAE_SCANNER_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Sirius requires its pinned tae-scanner submodule")
endif()
