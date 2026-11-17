"""Window-close support for the Linux RAI wheel used by the WSL demo."""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import robotic as ry


def make_close_checker(config: "ry.Config") -> Callable[[], bool]:
    """Return a close-request check after the first ``config.view(False)``.

    RAI 0.4.1 hides its window when its close button is pressed, without setting
    the key code returned by ``view``. Query the GLFW implementation exported
    by that same Linux wheel; loading a separate GLFW library would not share
    its state. No GUI or native library is loaded when this module is imported.

    Keep the configuration alive and call the checker only before closing the
    viewer. This helper does not handle keyboard events or destroy the window.
    """
    import ctypes
    from pathlib import Path
    import sys

    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Viewer close-button detection currently supports the Linux "
            "robotic==0.4.1 wheel. Run the demo in WSL Ubuntu or use --no-view."
        )

    import robotic as ry

    viewer = config.viewer()
    window = viewer.getGLFWWindow()
    if not window:
        raise RuntimeError(
            "Open the RAI viewer with config.view(False) before setting up "
            "close-button detection."
        )

    library_path = Path(ry.__file__).with_name("librai.so")
    try:
        library = ctypes.CDLL(str(library_path))
        should_close = library.glfwWindowShouldClose
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "The RAI wheel does not expose GLFW close-button detection. "
            "Install requirements-rai.txt in the project environment "
            "(Linux robotic==0.4.1), or use --no-view."
        ) from exc
    should_close.argtypes = [ctypes.c_void_p]
    should_close.restype = ctypes.c_int

    def close_requested() -> bool:
        # A replaced/closed viewer must not leave us querying its stale pointer.
        if viewer.getGLFWWindow() != window:
            return True
        return bool(should_close(window))

    return close_requested
