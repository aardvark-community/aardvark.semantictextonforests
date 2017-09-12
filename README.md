[![Join the chat at https://gitter.im/aardvark-platform/Lobby](https://img.shields.io/badge/gitter-join%20chat-blue.svg)](https://gitter.im/aardvark-platform/Lobby)
[![license](https://img.shields.io/github/license/aardvark-platform/aardvark.semantictextonforests.svg)](https://github.com/aardvark-platform/aardvark.semantictextonforests/blob/master/LICENSE)

[Wiki](https://github.com/aardvarkplatform/aardvark.docs/wiki) | 
[Gallery](https://github.com/aardvarkplatform/aardvark.docs/wiki/Gallery) | 
[Quickstart](https://github.com/aardvarkplatform/aardvark.docs/wiki/Quickstart-Windows) | 
[Status](https://github.com/aardvarkplatform/aardvark.docs/wiki/Status)

Aardvark.SemanticTextonForests is part of the open-source [Aardvark platform](https://github.com/aardvark-platform/aardvark.docs/wiki) for visual computing, real-time graphics and visualization.

## Aardvark.SemanticTextonForests

This projects contains
- a Semantic Texton Forests implementation in C# (http://www.matthewajohnson.org/research/stf.html)
- a standalone .NET wrapper for libsvm 3.20 (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

In order to ensure best possible performance, the libsvm wrapper is written in C++/CLI (for the .NET interface) and includes the original (untouched) C code of libsvm.
The project also includes x86 and x64 build targets for Visual Studio 2013 as well as Visual Studio 2015.

### How to build

**Windows:**
- Requirements: Visual Studio 2013 or 2015
- run build.cmd
- or build using Visual Studio with solution src/all.sln.

**Linux:**
- not yet supported

### Technical Report

Download at http://download.vrvis.at/acquisition/tr/VRVis_TR_szabo_maierhofer_semantictextonforests.pdf

If you use this code for any kind of publication please include the following reference:
```
@techreport{tr-vrvis-20150609,
     title = {{Implementation of Semantic Texton Forests for Image Categorization and Segmentation}},
     author = {Attila Szabo and Stefan Maierhofer},
     year = {2015},
     institution = {VRVis Research Center},
     month = {06},
     url = {http://download.vrvis.at/acquisition/tr/VRVis_TR_szabo_maierhofer_semantictextonforests.pdf}
}
```

### License

MIT (http://opensource.org/licenses/MIT)

Authors: Attila Szabo (aszabo@vrvis.at), Stefan Maierhofer (sm@vrvis.at)

Copyright © 2015 VRVis Zentrum für Virtual Reality und Visualisierung Forschungs-GmbH, Donau-City-Strasse 1, A-1220 Wien, Austria. http://www.vrvis.at.
