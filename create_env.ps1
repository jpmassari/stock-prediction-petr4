##Configurando projeto para o ENV
##Antes de tudo deve-se desabilitar o pip.ini do acesso global.
$executingScriptDirectory = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Write-Host ($executingScriptDirectory)
cd $executingScriptDirectory
virtualenv $executingScriptDirectory\env
.\env\Scripts\activate
.\env\Scripts\python.exe -m pip install --upgrade pip
pip install twine keyring artifacts-keyring
Copy-Item $executingScriptDirectory\pip.ini $executingScriptDirectory\env -Force
pip install -r requirements.txt