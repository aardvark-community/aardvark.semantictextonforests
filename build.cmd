@echo off
setlocal

set VSVERSION=140
set _msbuildexe="%ProgramFiles(x86)%\MSBuild\14.0\Bin\MSBuild.exe"
if not exist %_msbuildexe% (
    set _msbuildexe="%ProgramFiles(x86)%\MSBuild\12.0\Bin\MSBuild.exe"
	set VSVERSION=120
	)

PUSHD nuget
nuget restore ..\src\all.sln
POPD

PUSHD src
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Debug_VS2013";Platform="x64"
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Debug_VS2013";Platform="x86"
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Release_VS2013";Platform="x64"
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Release_VS2013";Platform="x86"
if "%VSVERSION%" == "140" (
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Debug_VS2015";Platform="x64"
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Debug_VS2015";Platform="x86"
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Release_VS2015";Platform="x64"
%_msbuildexe% all.sln /m /t:Build /p:Configuration="Release_VS2015";Platform="x86"
)
POPD

PUSHD nuget
nuget pack libsvm.clr.vs2013.x64.nuspec
nuget pack libsvm.clr.vs2013.x86.nuspec
nuget pack libsvm.clr.vs2013.x64.debug.nuspec
nuget pack libsvm.clr.vs2013.x86.debug.nuspec
if "%VSVERSION%" == "140" (
nuget pack libsvm.clr.vs2015.x64.nuspec
nuget pack libsvm.clr.vs2015.x86.nuspec
nuget pack libsvm.clr.vs2015.x64.debug.nuspec
nuget pack libsvm.clr.vs2015.x86.debug.nuspec
)
POPD

