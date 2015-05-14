REM nuget restore ../src/Uncodium.Perplexity.
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Debug_VS2013";Platform="x64";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Debug_VS2013";Platform="x86";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Debug_VS2015";Platform="x64";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Debug_VS2015";Platform="x86";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Release_VS2013";Platform="x64";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Release_VS2013";Platform="x86";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Release_VS2015";Platform="x64";SolutionDir=..\
msbuild ../libsvm.clr/libsvm.clr.vcxproj /m /t:Build /p:Configuration="Release_VS2015";Platform="x86";SolutionDir=..\
nuget pack libsvm.clr.vs2013.x64.nuspec
nuget pack libsvm.clr.vs2013.x86.nuspec
nuget pack libsvm.clr.vs2015.x64.nuspec
nuget pack libsvm.clr.vs2015.x86.nuspec
nuget pack libsvm.clr.vs2013.x64.debug.nuspec
nuget pack libsvm.clr.vs2013.x86.debug.nuspec
nuget pack libsvm.clr.vs2015.x64.debug.nuspec
nuget pack libsvm.clr.vs2015.x86.debug.nuspec