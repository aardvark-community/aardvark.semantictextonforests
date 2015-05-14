This projects contains
- a Semantic Texton Forests implementation in C# (http://www.matthewajohnson.org/research/stf.html)
- a standalone .NET wrapper for libsvm 3.20 (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

In order to ensure best possible performance, the libsvm wrapper is written in C++/CLI (for the .NET interface) and includes the original (untouched) C code of libsvm.
The project also includes x86 and x64 build targets for Visual Studio 2013 as well as Visual Studio 2015.


How to build:
=============

Windows:
- Requirements: Visual Studio 2013 or 2015
- run build.cmd
- or build using Visual Studio with solution src/all.sln.

Linux:
- not yet supported



License: http://opensource.org/licenses/MIT

Authors: Attila Szabo (aszabo@vrvis.at), Stefan Maierhofer (sm@vrvis.at)

Copyright © 2015 VRVis Zentrum für Virtual Reality und Visualisierung Forschungs-GmbH, Donau-City-Strasse 1, A-1220 Wien, Austria. http://www.vrvis.at.
