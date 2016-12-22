
file(GLOB MotionEstimationSources *.cpp *.cu)
file(GLOB MotionEstimationIncludes *.h )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

cuda_add_executable(motion-estimation ${MotionEstimationSources})
target_link_libraries(motion-estimation  gstreamer-1.0 gstapp-1.0 glib-2.0 gobject-2.0 Qt4::QtGui visionworks nvxio)

