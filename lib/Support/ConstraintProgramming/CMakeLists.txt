add_dynamatic_library(DynamaticConstraintProgramming
  ConstraintProgramming.cpp

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticSupportUtils
  DynamaticHW
  MLIRIR

  LINK_COMPONENTS
  Support
)

# The HiGHS backend links the solver into the library, so that no external
# executable is needed at runtime.
if(DYNAMATIC_ENABLE_HIGHS)
  target_link_libraries(DynamaticConstraintProgramming PUBLIC highs)
endif()
