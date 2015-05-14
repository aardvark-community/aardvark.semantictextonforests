PUSHD nuget
nuget.exe restore ..\libsvm.clr.sln
CALL buildNugetPackage.bat
POPD
