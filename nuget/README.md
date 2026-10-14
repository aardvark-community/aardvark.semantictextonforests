# Bundled NuGet CLI

`nuget.exe` is pinned to NuGet **6.14.3** (reported build: `6.14.3.1`).
It replaces NuGet 2.8, which is below Newtonsoft.Json 13.0.4's declared minimum
client version of 2.12.

- Source: <https://dist.nuget.org/win-x86-commandline/v6.14.3/nuget.exe>
- SHA-256: `8103c5666f63528d9fec59b53c5ae4b6feed7c8aec2930e344e870375f408a90`
- Authenticode signature: verified valid, Microsoft Corporation.
- Build-host runtime: **.NET Framework 4.7.2 or later**. The projects still target
  .NET Framework 4.5.

`build.cmd` restores the three `packages.config` manifests directly, avoiding a
requirement for a newer MSBuild just to discover packages. Restore failures stop
that script before compilation or packaging. Existing native compiler requirements
are unchanged.

To verify a replacement download before using it:

```powershell
Get-AuthenticodeSignature .\nuget\nuget.exe
Get-FileHash .\nuget\nuget.exe -Algorithm SHA256
```
