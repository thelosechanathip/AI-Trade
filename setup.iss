#define AppName    "AI-Trade"
#define AppVersion "2.3.0"
#define AppPublisher "AI-Trade"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=releases
OutputBaseFilename=AI-Trade_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
MinVersion=10.0
UninstallDisplayName={#AppName}
CloseApplications=no
DisableProgramGroupPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; ── Python source ────────────────────────────────────────────
Source: "ai_model.py";             DestDir: "{app}"; Flags: ignoreversion
Source: "auto_optimizer.py";       DestDir: "{app}"; Flags: ignoreversion
Source: "backtest.py";             DestDir: "{app}"; Flags: ignoreversion
Source: "dashboard.py";            DestDir: "{app}"; Flags: ignoreversion
Source: "execution_mt5.py";        DestDir: "{app}"; Flags: ignoreversion
Source: "main.py";                 DestDir: "{app}"; Flags: ignoreversion
Source: "market_memory.py";        DestDir: "{app}"; Flags: ignoreversion
Source: "monitor_improvements.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "risk.py";                 DestDir: "{app}"; Flags: ignoreversion
Source: "rl_agent.py";             DestDir: "{app}"; Flags: ignoreversion
Source: "run.py";                  DestDir: "{app}"; Flags: ignoreversion
Source: "strategy.py";             DestDir: "{app}"; Flags: ignoreversion
Source: "trade_manager.py";        DestDir: "{app}"; Flags: ignoreversion
Source: "utils.py";                DestDir: "{app}"; Flags: ignoreversion
Source: "web_app.py";              DestDir: "{app}"; Flags: ignoreversion

; ── Config (onlyifdoesntexist = ไม่ทับ config ที่แก้ไขแล้ว) ──
Source: "config.yaml";             DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist
Source: "config_safe.yaml";        DestDir: "{app}"; Flags: ignoreversion
Source: "config_aggressive.yaml";  DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt";        DestDir: "{app}"; Flags: ignoreversion

; ── Dashboard ────────────────────────────────────────────────
Source: "static\index.html"; DestDir: "{app}\static"; Flags: ignoreversion
Source: "dashboard\*"; DestDir: "{app}\dashboard"; \
  Flags: ignoreversion recursesubdirs createallsubdirs; \
  Excludes: "node_modules,.next"

; ── Launcher ─────────────────────────────────────────────────
Source: "start.bat"; DestDir: "{app}"; Flags: ignoreversion

; ── Empty placeholder dirs ───────────────────────────────────
Source: "data\.gitkeep";   DestDir: "{app}\data";   Flags: ignoreversion
Source: "models\.gitkeep"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "logs\.gitkeep";   DestDir: "{app}\logs";   Flags: ignoreversion

[Icons]
Name: "{autodesktop}\AI-Trade";      Filename: "{app}\start.bat"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\AI-Trade Engine";     Filename: "{app}\start.bat"; WorkingDir: "{app}"
Name: "{group}\Edit config.yaml";    Filename: "{app}\config.yaml"
Name: "{group}\Uninstall AI-Trade";  Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
; Create virtual environment
Filename: "{code:GetPythonPath}"; \
  Parameters: "-m venv ""{app}\venv"""; \
  WorkingDir: "{app}"; \
  StatusMsg: "Creating Python environment..."; \
  Flags: runhidden waituntilterminated

; Upgrade pip
Filename: "{app}\venv\Scripts\python.exe"; \
  Parameters: "-m pip install --upgrade pip -q"; \
  WorkingDir: "{app}"; \
  StatusMsg: "Preparing package manager..."; \
  Flags: runhidden waituntilterminated

; Install all packages
Filename: "{app}\venv\Scripts\pip.exe"; \
  Parameters: "install -r ""{app}\requirements.txt"""; \
  WorkingDir: "{app}"; \
  StatusMsg: "Installing packages (3-5 minutes, please wait)..."; \
  Flags: runhidden waituntilterminated

; Offer to launch
Filename: "{app}\start.bat"; \
  WorkingDir: "{app}"; \
  Description: "Launch AI-Trade now (make sure MT5 is open first)"; \
  Flags: nowait postinstall skipifsilent unchecked

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
var
  PythonPath: String;

{ ── Find Python 3.10+ via registry (HKCU then HKLM) ─────────}
function DetectPython(): String;
var
  SubKey, InstallPath, PyExe: String;
  Versions: TArrayOfString;
  i: Integer;
begin
  Result := '';

  SetArrayLength(Versions, 5);
  Versions[0] := '3.13';
  Versions[1] := '3.12';
  Versions[2] := '3.11';
  Versions[3] := '3.10';
  Versions[4] := '3.9';

  for i := 0 to High(Versions) do
  begin
    SubKey := 'SOFTWARE\Python\PythonCore\' + Versions[i] + '\InstallPath';

    if RegQueryStringValue(HKCU, SubKey, '', InstallPath) then
    begin
      PyExe := AddBackslash(InstallPath) + 'python.exe';
      if FileExists(PyExe) then begin Result := PyExe; Exit; end;
    end;

    if RegQueryStringValue(HKLM, SubKey, '', InstallPath) then
    begin
      PyExe := AddBackslash(InstallPath) + 'python.exe';
      if FileExists(PyExe) then begin Result := PyExe; Exit; end;
    end;

    { 64-bit Python on 32-bit installer }
    SubKey := 'SOFTWARE\WOW6432Node\Python\PythonCore\' + Versions[i] + '\InstallPath';
    if RegQueryStringValue(HKLM, SubKey, '', InstallPath) then
    begin
      PyExe := AddBackslash(InstallPath) + 'python.exe';
      if FileExists(PyExe) then begin Result := PyExe; Exit; end;
    end;
  end;

  { Last resort: common fixed paths }
  if FileExists(ExpandConstant('{localappdata}\Programs\Python\Python311\python.exe')) then
    Result := ExpandConstant('{localappdata}\Programs\Python\Python311\python.exe')
  else if FileExists(ExpandConstant('{localappdata}\Programs\Python\Python310\python.exe')) then
    Result := ExpandConstant('{localappdata}\Programs\Python\Python310\python.exe');
end;

function GetPythonPath(Param: String): String;
begin
  Result := PythonPath;
end;

function InitializeSetup(): Boolean;
var
  Answer: Integer;
begin
  Result := True;
  PythonPath := DetectPython();

  if PythonPath = '' then
  begin
    Answer := MsgBox(
      'Python 3.10+ is required but was not found.' + Chr(13) + Chr(10) +
      Chr(13) + Chr(10) +
      'Please install Python from:' + Chr(13) + Chr(10) +
      'https://www.python.org/downloads/' + Chr(13) + Chr(10) +
      Chr(13) + Chr(10) +
      'Check "Add Python to PATH" during installation.' + Chr(13) + Chr(10) +
      'Click OK to open the download page.',
      mbError, MB_OKCANCEL);
    if Answer = IDOK then
      ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOW, ewNoWait, Answer);
    Result := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(
      'AI-Trade installed successfully!' + Chr(13) + Chr(10) +
      Chr(13) + Chr(10) +
      'Before running:' + Chr(13) + Chr(10) +
      '1. Open MetaTrader 5 and log in' + Chr(13) + Chr(10) +
      '2. Edit config.yaml to set your broker symbol' + Chr(13) + Chr(10) +
      Chr(13) + Chr(10) +
      'Then double-click AI-Trade on your Desktop.',
      mbInformation, MB_OK);
  end;
end;
