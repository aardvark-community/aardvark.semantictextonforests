# Dependabot remediation and verification

## Scope

Alerts #4–#6 all report CVE-2024-21907 / GHSA-5crp-9r3c-p9vr in Newtonsoft.Json
6.0.8. All three manifests and project references now select **13.0.4**, including
its `lib/net45` assembly. Other dependency versions, framework targets, native
toolsets, and `TypeNameHandling.All` are unchanged. Obsolete
`TypeNameAssemblyFormat` assignments use the equivalent
`TypeNameAssemblyFormatHandling.Full` API.

The [bundled NuGet client](../nuget/README.md) was updated to pinned 6.14.3.
`build.cmd` restores manifests without newer-MSBuild discovery and stops on
restore failure. No binding redirects were needed in the tested configurations.

## Checks performed

- Original four xUnit tests passed before changes.
- [Legacy fixtures](../src/Tests/Fixtures/README.md) were captured with 6.0.8.
- Three bounded excessive-nesting tests failed with 6.0.8 because all three
  loaders accepted the input without an exception. They pass with 13.0.4,
  rejecting it with `JsonReaderException` at the default finite depth limit.
- A clean direct restore of all three updated manifests completed without
  vulnerability warnings or MSBuild discovery. The restored package set contains
  only Newtonsoft.Json 13.0.4. An isolated missing-manifest check confirmed that
  `build.cmd` exits with code 1 before compilation or packaging.
- All eight Debug/Release × VS2013/VS2015 configuration names × x86/x64 builds
  passed **with a local v142 compiler override**, Windows SDK 10, and .NET 4.5
  reference assemblies from `Microsoft.NETFramework.ReferenceAssemblies.net45`
  1.0.3. No project retargeting was committed.
- Each diagnostic configuration ran 12 xUnit cases: **11 passed, 1 skipped**.
  Coverage includes the original native-wrapper tests, legacy mutable forest
  graph/polymorphic provider imports, empty image sets, test results, round trips,
  a 25-level forest (matching an example), and all three nesting-rejection paths.
- All 22 output Newtonsoft.Json DLL hashes match the restored 13.0.4 net45 DLL.
- All 16 existing native-wrapper nuspecs were locally packed with the new client.
  It reported legacy license/dependency-group/readme metadata warnings. These
  diagnostic binaries/packages are not certified release artifacts; nothing was
  published.

## Reproduction

On a machine with the documented VS2013/VS2015 prerequisites, run `build.cmd`.
For the VS2015 x64 Release test output:

```powershell
.\src\packages\xunit.runner.console.2.0.0\tools\xunit.console.exe `
    .\src\Tests\bin\x64\Release_2015\Tests.exe -noshadow
```

Local diagnostic builds used VS2019 Build Tools MSBuild and these extra
command-line properties, not project-file changes:

```powershell
.\nuget\nuget.exe install Microsoft.NETFramework.ReferenceAssemblies.net45 `
    -Version 1.0.3 -OutputDirectory src/build/dependabot/tools `
    -Source https://api.nuget.org/v3/index.json -NonInteractive

$msbuild = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
$references = "$PWD/src/build/dependabot/tools/Microsoft.NETFramework.ReferenceAssemblies.net45.1.0.3/build/"
& $msbuild src/all.sln /m /t:Rebuild /p:Configuration=Release_VS2015 `
    /p:Platform=x64 /p:PlatformToolset=v142 /p:WindowsTargetPlatformVersion=10.0 `
    "/p:TargetFrameworkRootPath=$references"
```

Repeat for `Debug_VS2013`, `Release_VS2013`, `Debug_VS2015`, and
`Release_VS2015`, with both platforms. Use `xunit.console.x86.exe` for x86;
the existing solution maps its x86 Tests configurations to `bin/x86/Debug`.
The other x64 Tests outputs are `bin/x64/<configuration>`.

## Limits and follow-ups

- All eight attempts with the original toolsets were blocked by missing v120 or
  v140 build tools (`MSB8020`). Diagnostic success is **not** certification with
  those native compilers. Their matrix remains to be verified on a suitable host.
- The skipped test and forest metadata limitation reproduce with the old package;
  see the [saved-model round-trip proposal](ideas/json-round-trip-data-loss.md).
  This work does not claim full saved-model compatibility.
- The package upgrade does not secure unrestricted JSON type-name resolution for
  untrusted files. See the [trust-boundary proposal](ideas/json-deserialization-trust-boundary.md).
- Once these changes reach `master` and GitHub rescans them, verify that alerts
  #4–#6 become **fixed**, not dismissed.
