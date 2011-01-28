; -- refl1d.iss -- an Inno Setup Script for Refl1d
; This script is used by the Inno Setup Compiler to build a Windows XP
; installer/uninstaller for the DANSE Reflectometry application named Refl1d.
; The script is written to explicitly allow multiple versions of the
; application to be installed simulaneously in separate subdirectories such
; as "Refl1d 0.5.0", "Refl1d 0.7.2", and "Refl1d 1.0" under a group directory.

; NOTE: In order to support more than one version of the application
; installed simultaneously, the AppName, Desktop shortcut name, and Quick
; Start shortcut name must be unique among versions.  This is in addition to
; having unique names (in the more obvious places) for DefaultDirNam,
; DefaultGroupName, and output file name.

; By default, when installing Refl1d:
; - The destination folder will be "C:\Program Files\DANSE\Refl1d x.y.z"
; - A desktop icon will be created with the label "Refl1d x.y.z"
; - A quickstart icon is optional
; - A start menu folder will be created with the name DANSE -> Refl1d x.y.z
; By default, when uninstalling Refl1d x.y.z
; - The uninstall can be initiated from either the:
;   * Start menu via DANSE -> Refl1d x.y.z -> Uninstall Refl1d
;   * Start menu via Control Panel - > Add or Remove Programs -> Refl1d x.y.z
; - It will not delete the C:\Program Files\DANSE\Refl1d x.y.z folder if it
;   contains any user created files
; - It will delete any desktop or quickstart icons for Refl1d that were
;   created on installation

; NOTE: The Quick Start Pack for the Inno Setup Compiler needs to be installed
; with the Preprocessor add-on selected to support use of #define statements.
#define MyAppName "Refl1d"
#define MyAppNameLowercase "refl1d"
#define MyAppVersion "0.0.0"
#define MyGroupFolderName "DANSE"
#define MyAppPublisher "NIST & University of Maryland"
#define MyAppURL "http://www.reflectometry.org/danse/"
#define MyAppFileName "refl1d.exe"
#define MyIconFileName "refl1d.ico"
#define MyReadmeFileName "README.txt"
#define MyLicenseFileName "LICENSE.txt"
#define Space " "
; Use updated version string if present in the include file.  It is expected that the
; Refl1d build script will create this file using the version string from version.py.
#ifexist "refl1d.iss-include"
    #include "refl1d.iss-include"
#endif

[Setup]
; Make the AppName string unique so that other versions of the program can be installed simultaniously.
; This is done by using the name and version of the application together as the AppName.
AppName={#MyAppName}{#Space}{#MyAppVersion}
AppVerName={#MyAppName}{#Space}{#MyAppVersion}
AppPublisher={#MyAppPublisher}
ChangesAssociations=yes
; If you do not want a space in folder names, omit {#Space} or replace it with a hyphen char, etc.
DefaultDirName={pf}\{#MyGroupFolderName}\{#MyAppName}{#Space}{#MyAppVersion}
DefaultGroupName={#MyGroupFolderName}\{#MyAppName}{#Space}{#MyAppVersion}
Compression=lzma/max
SolidCompression=yes
DisableProgramGroupPage=yes
; A file extension of .exe will be appended to OutputBaseFilename.
OutputBaseFilename={#MyAppNameLowercase}-{#MyAppVersion}-win32
OutputManifestFile={#MyAppNameLowercase}-{#MyAppVersion}-win32-manifest.txt
SetupIconFile={#MyIconFileName}
LicenseFile={#MyLicenseFileName}
SourceDir=.
OutputDir=.
PrivilegesRequired=none
;;;InfoBeforeFile=display_before_install.txt
;;;InfoAfterFile=display_after_install.txt

; The App*URL directives are for display in the Add/Remove Programs control panel and are all optional
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; This script assumes that the output from the previously run py2exe packaging process is in .\dist\...
; NOTE: Don't use "Flags: ignoreversion" on any shared system files
Source: "dist\*"; Excludes: "examples,doc"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "dist\examples\*"; DestDir: "{userdocs}\{#MyAppName}\examples"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "dist\doc\*"; DestDir: "{userdocs}\{#MyAppName}\doc"; Flags: ignoreversion recursesubdirs createallsubdirs

; The following Pascal function checks for the presence of the VC++ 2008 DLL folder on the target system
; to determine if the VC++ 2008 Redistributable kit needs to be installed.
[Code]
function InstallVC90CRT(): Boolean;
begin
    Result := not DirExists('C:\WINDOWS\WinSxS\x86_Microsoft.VC90.CRT_1fc8b3b9a1e18e3b_9.0.21022.8_x-ww_d08d0375');
end;

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Icons]
; This section does not have a flag for keeping the command window open on exit when running a program.
; To accomplish this, a batch file is run that creates the command window and starts the Windows command
; interpreter.  This provides the same environment as starting a command window using the run dialog box
; from the Windows start menu and entering a command such as "cmd" or "cmd /k <file-to-execute>".
;;;Name: "{group}\Launch {#MyAppName}"; Filename: "{app}\{#MyAppFileName}"; IconFilename: "{app}\{#MyIconFileName}"; WorkingDir: "{userdocs}\{#MyAppName}"; Flags: runmaximized
Name: "{group}\Launch {#MyAppName}"; Filename: "{app}\refllaunch.bat"; IconFilename: "{app}\{#MyIconFileName}"; WorkingDir: "{userdocs}\{#MyAppName}"; Flags: runmaximized
Name: "{group}\{cm:ProgramOnTheWeb,{#MyAppName}}"; Filename: "{#MyAppURL}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
;;;Name: "{commondesktop}\{#MyAppName}{#Space}{#MyAppVersion}"; Filename: "{app}\{#MyAppFileName}"; Tasks: desktopicon; WorkingDir: "{userdocs}\{#MyAppName}"; IconFilename: "{app}\{#MyIconFileName}"; Flags: runmaximized
Name: "{commondesktop}\{#MyAppName}{#Space}{#MyAppVersion}"; Filename: "{app}\refllaunch.bat"; Tasks: desktopicon; WorkingDir: "{userdocs}\{#MyAppName}"; IconFilename: "{app}\{#MyIconFileName}"; Flags: runmaximized
;;;Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}{#Space}{#MyAppVersion}"; Filename: "{app}\{#MyAppFileName}"; Tasks: quicklaunchicon; WorkingDir: "{userdocs}\{#MyAppName}"; IconFilename: "{app}\{#MyIconFileName}"; Flags: runmaximized
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}{#Space}{#MyAppVersion}"; Filename: "{app}\refllaunch.bat"; Tasks: quicklaunchicon; WorkingDir: "{userdocs}\{#MyAppName}"; IconFilename: "{app}\{#MyIconFileName}"; Flags: runmaximized

[Run]
;;;Filename: "{app}\{#MyAppFileName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; WorkingDir: "{userdocs}\{#MyAppName}"; Flags: nowait postinstall skipifsilent runmaximized
Filename: "{app}\refllaunch.bat"; Description: "{cm:LaunchProgram,{#MyAppName}}"; WorkingDir: "{userdocs}\{#MyAppName}"; Flags: nowait postinstall skipifsilent runmaximized
Filename: "{app}\{#MyReadmeFileName}"; Description: "Read Release Notes"; Verb: "open"; Flags: shellexec skipifdoesntexist waituntilterminated postinstall skipifsilent unchecked
; Install the Microsoft C++ DLL redistributable package if it is provided and the DLLs are not present on the target system.
; Note that the redistributable package is included if the app was built using Python 2.6 or 2.7, but not with 2.5.
; Parameter options:
; - for silent install use: "/q"
; - for silent install with progress bar use: "/qb"
; - for silent install with progress bar but disallow cancellation of operation use: "/qb!"
; Note that we do not use the postinstall flag as this would display a checkbox and thus require the user to decide what to do.
Filename: "{app}\vcredist_x86.exe"; Parameters: "/qb!"; WorkingDir: "{tmp}"; StatusMsg: "Installing Microsoft Visual C++ 2008 Redistributable Package ..."; Check: InstallVC90CRT(); Flags: skipifdoesntexist waituntilterminated

[UninstallDelete]
; Delete directories and files that are dynamically created by the application.
Type: filesandordirs; Name: "{app}\.matplotlib"

