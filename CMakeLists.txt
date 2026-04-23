cmake_minimum_required(VERSION 3.15)
project(rl_rubrics C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(rubrics
    src/main.c
    src/gl_loader.c
    src/cube.c
    src/camera.c
    src/renderer.c
)

target_include_directories(rubrics PRIVATE src)
target_link_libraries(rubrics PRIVATE glfw OpenGL::GL)

# Headless trainer. Shares cube.c with the frontend; no graphics deps.
add_executable(train
    src/train.c
    src/cube.c
    src/nn.c
    src/rng.c
)
target_include_directories(train PRIVATE src)

# GPU compute-shader sanity check: confirms the system supports
# OpenGL 4.3 compute and the SSBO/dispatch pipeline works end-to-end.
add_executable(gpu_check
    src/gpu_check.c
    src/gl_loader.c
)
target_include_directories(gpu_check PRIVATE src)
target_link_libraries(gpu_check PRIVATE glfw OpenGL::GL)
if(WIN32)
    target_link_libraries(gpu_check PRIVATE gdi32)
endif()

# GPU MLP correctness + throughput test: same weights through CPU MLP
# and GpuMlp, verifies outputs match within float precision.
add_executable(gpu_nn_test
    src/gpu_nn_test.c
    src/gpu_nn.c
    src/gl_loader.c
    src/nn.c
    src/rng.c
)
target_include_directories(gpu_nn_test PRIVATE src)
target_link_libraries(gpu_nn_test PRIVATE glfw OpenGL::GL)
if(WIN32)
    target_link_libraries(gpu_nn_test PRIVATE gdi32)
endif()

# Batched GPU trainer: B parallel episodes, GPU forward, CPU backward.
add_executable(gpu_train
    src/gpu_train.c
    src/gpu_nn.c
    src/gl_loader.c
    src/cube.c
    src/nn.c
    src/rng.c
)
target_include_directories(gpu_train PRIVATE src)
target_link_libraries(gpu_train PRIVATE glfw OpenGL::GL)
if(WIN32)
    target_link_libraries(gpu_train PRIVATE gdi32)
endif()

# GPU-backed visualizer: same batched training as gpu_train, with the
# 3D cube renderer + HUD on top. Visualizes episode 0.
add_executable(gpu_train_viz
    src/gpu_train_viz.c
    src/gpu_nn.c
    src/gl_loader.c
    src/cube.c
    src/camera.c
    src/renderer.c
    src/hud.c
    src/nn.c
    src/rng.c
)
target_include_directories(gpu_train_viz PRIVATE src)
target_link_libraries(gpu_train_viz PRIVATE glfw OpenGL::GL)
if(WIN32)
    target_link_libraries(gpu_train_viz PRIVATE gdi32)
endif()

# Live visualization of training progress: cube + HUD bars.
add_executable(train_viz
    src/train_viz.c
    src/cube.c
    src/camera.c
    src/renderer.c
    src/gl_loader.c
    src/hud.c
    src/nn.c
    src/rng.c
)
target_include_directories(train_viz PRIVATE src)
target_link_libraries(train_viz PRIVATE glfw OpenGL::GL)
if(WIN32)
    target_link_libraries(train_viz PRIVATE gdi32)
endif()
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(train_viz PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(WIN32)
    target_link_libraries(rubrics PRIVATE gdi32)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rubrics PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# On Windows, copy glfw3.dll next to the exe if it is a shared library
# so the binary runs without needing msys64/mingw64/bin on PATH.
if(WIN32)
    get_target_property(_glfw_type glfw TYPE)
    if(_glfw_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(TARGET rubrics POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:glfw>" "$<TARGET_FILE_DIR:rubrics>"
            VERBATIM)
    else()
        # Imported target from MSYS2 may not expose SHARED_LIBRARY;
        # fall back to probing the standard mingw64 path.
        set(_glfw_dll "C:/msys64/mingw64/bin/glfw3.dll")
        if(EXISTS "${_glfw_dll}")
            add_custom_command(TARGET rubrics POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_glfw_dll}" "$<TARGET_FILE_DIR:rubrics>"
                VERBATIM)
        endif()
    endif()
endif()
