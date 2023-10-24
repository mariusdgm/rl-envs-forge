echo off
setlocal enabledelayedexpansion

REM Set the root directory
set "ROOT_DIR=%~dp0..\.."

REM Change to rl_envs_forge\envs directory to traverse its subdirectories
pushd %ROOT_DIR%\rl_envs_forge\envs

@REM echo Current Directory: %CD%

FOR /D %%i IN (*) DO (
    if "%%i"=="__pycache__" (
        echo Skipping %%i
    ) else (
        REM Check if the directory contains __init__.py
        if exist "%%i\__init__.py" (

            REM Generate diagrams specific to the environment
            pushd %ROOT_DIR%
            pyreverse -o png -p %%i rl_envs_forge.envs.%%i

            REM Ensure destination directory exists
            if not exist "%ROOT_DIR%\docs\diagrams\%%i" mkdir "%ROOT_DIR%\docs\diagrams\%%i"
            
            REM Move the generated diagrams to the specific environment's folder
            move classes_%%i.png "%ROOT_DIR%\docs\diagrams\%%i\"
            move packages_%%i.png "%ROOT_DIR%\docs\diagrams\%%i\"

            popd
        ) else (
            echo Skipping %%i as it does not contain __init__.py
        )
    )
)

popd

cd %ROOT_DIR%
echo Diagrams have been generated and moved to the diagrams folder ("%ROOT_DIR%\docs\diagrams\").
@REM pause