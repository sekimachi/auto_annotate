from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": [],
    "excludes": []
}

setup(
    name="myapp",
    version="1.0",
    description="My Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("auto_annotate.py")]
)