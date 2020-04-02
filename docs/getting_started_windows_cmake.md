# Getting Started on Windows with CMake

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_linux_cmake.md.
      Please keep them in sync.
-->

## Prerequisites

### Install CMake

Install CMake version >= 3.13 from https://cmake.org/download/.

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Your editor of choice likely has plugins for CMake,
> such as the Visual Studio Code
> [CMake Tools](https://github.com/microsoft/vscode-cmake-tools) extension.

### Install a Compiler

We recommend MSVC from either the full Visual Studio or from "Build Tools For
Visual Studio":

*   Choose either option from the
    [downloads page](https://visualstudio.microsoft.com/downloads/) and during
    installation make sure you include "C++ Build Tools"
*   Check that MSBuild is on your PATH. The path typically looks like:<br>
    `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin`

## Clone and Build

### Clone

Using your shell of choice (such as PowerShell or [cmder](https://cmder.net/)),
clone the repository and initialize its submodules:

```shell
$ git clone https://github.com/google/iree.git
$ cd iree
$ git submodule update --init
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Configure:

```shell
$ cmake -B build\ .
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;The root
> [CMakeLists.txt](https://github.com/google/iree/blob/master/CMakeLists.txt) file
> has options for configuring which parts of the project to enable.

Build all targets:

```shell
$ cmake --build build\ -j 8
```

## What's next?

### Take a Look Around

Check out the contents of the 'tools' build directory:

```shell
$ dir build\iree\tools\Debug
$ .\build\iree\tools\Debug\iree-translate.exe --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/master/iree/tools/test/simple.mlir)
and execute a function in the compiled module:

```shell
$ .\build\iree\tools\Debug\iree-run-mlir.exe %cd%/iree/tools/test/simple.mlir -input-value="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

More documentation coming soon...

<!-- TODO(scotttodd): Vulkan / other driver configuration -->
<!-- TODO(scotttodd): Running tests -->
<!-- TODO(scotttodd): Running samples -->
<!-- TODO(scotttodd): "getting_started.md" equivalent for iree-translate etc. -->